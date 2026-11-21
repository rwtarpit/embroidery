// TF32 GEMM for A100 (sm_80): C[M,N] = A[M,K] * B[K,N], row-major.
// Requires M % 256 == 0, N % 128 == 0, K % 32 == 0, K >= 32*(STAGES-1).

#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma_tf32(float (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__(256)
GEMM_tc(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int N, int M, int K)
{
    static_assert(BM == 256 && BN == 128 && BK == 32, "swizzles / warp layout assume 256x128x32");
    static_assert(STAGES >= 2, "need at least 2 stages");
    constexpr int WM = 64, WN = 64;          // warp tile, 4 (m) x 2 (n) warps
    constexpr int MT = WM / 16;              // 4  m16 tiles
    constexpr int NP = WN / 16;              // 4  pairs of n8 tiles
    constexpr int KT = BK / 8;               // 4  k8 steps per tile
    constexpr int A_STAGE = BM * BK;         // floats
    constexpr int B_STAGE = BK * BN;         // floats

    // ---- grid swizzle (unchanged) ----
    const int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    constexpr int SWIZZLE_W = 8;
    const int tile_col = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    const int tile_row = (linear_id / SWIZZLE_W) % gridDim.y;

    extern __shared__ __align__(128) float smem[];
    float* Bs = smem + STAGES * A_STAGE;
    const uint32_t As_s = smem_u32(smem);
    const uint32_t Bs_s = smem_u32(Bs);

    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tig = lane & 3;
    const int warp_m = warp_id & 3, warp_n = warp_id >> 2;

    // ---- gmem -> smem (cp.async) addressing ----
    const int a_row = tid >> 3,  a_ch = tid & 7;     // 8 threads x 16B per A row, 32 rows / pass
    const int b_row = tid >> 5,  b_ch = tid & 31;    // 32 threads x 16B per B row, 8 rows / pass
    // A: 16B chunk c of row r lives at chunk c ^ (r&7)   (r&7 is constant over passes: stride 32)
    // B: 16B chunk c of row r lives at chunk c ^ ((r&3)<<1) (stride 8 -> r&3 constant over passes)
    const uint32_t a_st = (a_row * 32  + ((a_ch ^ (a_row & 7)) << 2)) * 4;
    const uint32_t b_st = (b_row * 128 + ((b_ch ^ ((b_row & 3) << 1)) << 2)) * 4;
    const float* a_g = A + (size_t)(tile_row * BM + a_row) * K + a_ch * 4;
    const float* b_g = B + (size_t)b_row * N + tile_col * BN + b_ch * 4;
    const size_t a_rs = (size_t)32 * K;
    const size_t b_rs = (size_t)8 * N;
    const size_t b_ks = (size_t)BK * N;

    auto load_tile = [&](int stage) {
        const uint32_t a_dst = As_s + stage * (A_STAGE * 4) + a_st;
        const uint32_t b_dst = Bs_s + stage * (B_STAGE * 4) + b_st;
#pragma unroll
        for (int it = 0; it < BM / 32; ++it) cp_async16(a_dst + it * 4096, a_g + it * a_rs);
#pragma unroll
        for (int it = 0; it < BK / 8; ++it)  cp_async16(b_dst + it * 4096, b_g + it * b_rs);
        a_g += BK;
        b_g += b_ks;
    };

    const int NUM_TILES = K / BK;

    // ---- prologue: STAGES-1 tiles in flight ----
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < NUM_TILES) load_tile(s);
        cp_async_commit();
    }

    // ---- smem -> register fragment addressing, hoisted ----
    // A (ldmatrix.x4): lane -> row (lane&15), k-half (lane>>4); row&7 == lane&7
    uint32_t a_off[KT];
    {
        const int kh = lane >> 4;
        const uint32_t rowb = (warp_m * WM + (lane & 15)) * 128;
#pragma unroll
        for (int it = 0; it < KT; ++it)
            a_off[it] = rowb + ((((2 * it + kh) ^ (lane & 7))) << 4);
    }
    // B (LDS.64): thread reads physical columns {16p+2g, 16p+2g+1} of rows tig and tig+4.
    //   .x -> B-fragment of n8-tile 2p, .y -> B-fragment of n8-tile 2p+1  (n permutation)
    int b_off[NP];   // in floats
    {
        const int bc = (warp_n * 16 + (g >> 1)) ^ (tig << 1);   // swizzled 16B-chunk index
#pragma unroll
        for (int p = 0; p < NP; ++p)
            b_off[p] = tig * 128 + ((bc ^ (4 * p)) << 2) + ((g & 1) << 1);
    }

    uint32_t fa[2][MT][4];
    float2   fb[2][NP][2];
    float    acc[MT][2 * NP][4] = {};

    auto load_frags = [&](uint32_t (&xa)[MT][4], float2 (&xb)[NP][2], int st, int it) {
        const uint32_t a_addr = As_s + st * (A_STAGE * 4) + a_off[it];
#pragma unroll
        for (int i = 0; i < MT; ++i) ldmatrix_x4(xa[i], a_addr + i * 2048);
        const float* bp = Bs + st * B_STAGE + it * 1024;
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            xb[p][0] = *reinterpret_cast<const float2*>(bp + b_off[p]);         // k = 8*it + tig
            xb[p][1] = *reinterpret_cast<const float2*>(bp + b_off[p] + 512);   // k = 8*it + tig + 4
        }
    };
    auto mma_all = [&](uint32_t (&xa)[MT][4], float2 (&xb)[NP][2]) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                mma_tf32(acc[i][2 * p],     xa[i], __float_as_uint(xb[p][0].x), __float_as_uint(xb[p][1].x));
                mma_tf32(acc[i][2 * p + 1], xa[i], __float_as_uint(xb[p][0].y), __float_as_uint(xb[p][1].y));
            }
        }
    };

    // tile 0 must have landed
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    load_frags(fa[0], fb[0], 0, 0);

    int rd = 0, wr = STAGES - 1;
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        const int rd_next = (rd + 1 == STAGES) ? 0 : rd + 1;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            if (kk == KT - 1) {
                // tile+1 must have landed; also guarantees every warp finished reading stage (tile-1)
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                load_frags(fa[(kk + 1) & 1], fb[(kk + 1) & 1], rd_next, 0);
            } else {
                load_frags(fa[(kk + 1) & 1], fb[(kk + 1) & 1], rd, kk + 1);
            }
            if (kk == 0) {
                // stage 'wr' == (tile-1)%STAGES: all its readers are past the previous barrier
                if (tile + STAGES - 1 < NUM_TILES) load_tile(wr);
                cp_async_commit();   // always commit (possibly empty) so group counting stays uniform
            }
            mma_all(fa[kk & 1], fb[kk & 1]);
        }
        rd = rd_next;
        wr = (wr + 1 == STAGES) ? 0 : wr + 1;
    }

    // ---- epilogue: thread owns 4 contiguous columns -> float4 stores ----
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            const int r = tile_row * BM + warp_m * WM + i * 16 + g;
            const int c = tile_col * BN + warp_n * WN + p * 16 + tig * 4;
            float* dst = C + (size_t)r * N + c;
            // physical col offsets: +0 E.c0, +1 O.c0, +2 E.c1, +3 O.c1   (E = tile 2p, O = tile 2p+1)
            *reinterpret_cast<float4*>(dst) =
                make_float4(acc[i][2 * p][0], acc[i][2 * p + 1][0], acc[i][2 * p][1], acc[i][2 * p + 1][1]);
            *reinterpret_cast<float4*>(dst + (size_t)8 * N) =
                make_float4(acc[i][2 * p][2], acc[i][2 * p + 1][2], acc[i][2 * p][3], acc[i][2 * p + 1][3]);
        }
    }
}

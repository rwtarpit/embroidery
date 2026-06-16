/*
uses 128x16x128 strictly and 4096^3 GEMM
achieves 87% on A100 even when tuned for RTX consumer card.

*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define UINT2(value) (reinterpret_cast<uint2 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

const int WARP_SIZE = 32;



#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((((((row) >> 1)) ^ ((row) >> 3))) & 0x3) << 2))

#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- Inline PTX Assembly Macros ----------------
// cp.async: async copy 16 bytes from gmem (src) to smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                                \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                                                 \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))



// a block calculate c[128][128]
template<uint BM, uint BN, uint BK, uint NUM_THREADS, uint ACCUM_SIZE, uint NUM_STAGES>
__global__ void __launch_bounds__(NUM_THREADS) GEMM_tc(float* a, float* b, float* c, long long* dbg, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // grid swizzle tile width

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    // double buffer
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BM];

    // 2x4 warp tiling
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};
    // ------------------------------------------- Prefetch first tile
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];
    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&Bs[0][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

    float *global_b0 = &b[(0 + load_b_row) * n + bx * BN + load_b_col];
    float *global_b1 = &b[(0 + load_b_row + 8) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // main loop
    for (int bk = BK; bk < k; bk += BK) {

        // 1. Issue async prefetch of next A tile into write_idx buffer
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];
        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // cp.async load B
        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

        float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute using current read_idx buffer
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4×16: 64 rows in M dimension
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }

        // 3. cp.async sync
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // Swap buffers
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // Process last prefetched tile
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int n_base = warp_id_n * 32 + n_idx * 8;
            int b_col = n_base + (lane_id / 4);
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;

            reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
            reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
                        sum[m_idx][n_idx][1],
                        sum[m_idx][n_idx][2],
                        sum[m_idx][n_idx][3],
                        reg_a[m_idx][0],
                        reg_a[m_idx][1],
                        reg_a[m_idx][2],
                        reg_a[m_idx][3],
                        reg_b[n_idx][0],
                        reg_b[n_idx][1]);
            }
        }
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
    // Reuse As/Bs for coalesced C write-back (tricky)
    /**
    float (*Cs)[128] = (float (*)[128])(&As[0][0][0]);

    #define SWIZZLE_C(row, col) ((col) ^ (((row) << 3) & 127))

    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        __syncthreads();

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int smem_row = warp_id_m * 16 + t_row;
            int smem_col = warp_id_n * 32 + n_idx * 8 + t_col;

            FLOAT2(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(Cs[smem_row + 8][SWIZZLE_C(smem_row + 8, smem_col)]) = FLOAT2(sum[m_idx][n_idx][2]);
        }

        __syncthreads();

        int t_c_row = tid / 32;
        int t_c_col = (tid % 32) * 4;

#pragma unroll
        for (int step = 0; step < 4; ++step) {
            int smem_row = t_c_row + step * 8;
            int smem_col = t_c_col;

            float4 res = FLOAT4(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]);

            // Recover physical coordinates
            int global_row = by * BM + m_idx * 16 + (smem_row < 16 ? smem_row : 64 + smem_row - 16);
            int global_col = bx * BN + smem_col;

            FLOAT4(c[global_row * n + global_col]) = res;
        }
    }*/
}

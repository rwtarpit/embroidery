import cutlass
import torch
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda
import math
# beats cublas; cublas - ~1550 TFLOPS, this kernel ~1580 for 4096^3

class GEMM():
    def __init__(
        self,
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        mma_promotion_interval: int,
        max_active_clusters: int
    ):
        self.acc_dtype = cutlass.Float32
        self.mma_promotion_interval = mma_promotion_interval
        self.swizzle_size = swizzle_size
        self.raster_among_m = raster_along_m
        self.tile_shape_mn = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.max_active_clusters = max_active_clusters
        
        self.mma_layout_mnk = (2,1,1)
        self.num_wg_mma = math.prod(self.mma_layout_mnk)
        self.num_wg_dma = 1
        self.num_warps_per_wg = 4
        self.threads_per_wg = self.num_warps_per_wg * 32
        self.threads_per_cta = (self.num_wg_dma + self.num_wg_mma) * self.threads_per_wg
        self.registers_dma = 24
        self.registers_mma = 240
        
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.num_stages = 4
        self.epi_num_stages = 2
        self.epi_store_warp_id = 1
        self.num_mma_threads = (
            self.num_wg_mma * self.threads_per_wg
        )
        
        self.buffer_align_bytes = 1024
        
        
    def _setup_attributes(self):
        
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.mma_layout_mnk,
            (64, self.tile_shape_mn[1])
        )
        
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mn[0],
            self.tile_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        #num_k_blocks = mma_inst_tile_k
        #if self.mma_promotion_interval % mma_inst_tile_k != 0:
        #    raise ValueError(
        #        f"mma_promotion_interval ({self.mma_promotion_interval}) must be a "
        #        f"multiple of num_k_blocks ({mma_inst_tile_k})"
        #    )
        #self.tile_shape_mnk = (self.tile_shape_mn, num_k_blocks)
        is_cooperative = self.mma_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative, epi_tile_override=(128, 64)
        )
        self.cluster_layout_mn = cute.make_layout((self.cluster_shape_mn))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_layout,
            self.a_dtype,
            self.b_layout,
            self.b_dtype,
            self.c_layout,
            self.c_dtype,
            self.num_stages,
            self.epi_num_stages
        )
        
        
    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_layout: utils.LayoutEnum,
        a_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        c_dtype: type[cutlass.Numeric],
        num_stages: int,
        epi_num_stages: int
    ):
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, num_stages),
            order=(1, 0, 2) if a_is_k_major else (0, 1, 2),
        )
        
        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        )
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, num_stages),
            order=(1, 0, 2) if b_is_k_major else (0, 1, 2),
        )   # b tile is (n,k) row major
        
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size
            ),
            c_dtype
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(epi_tile, epi_num_stages),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2)
        )
        
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged
    
    
    @staticmethod
    def _make_tma_atom_and_tensor(
        gmem_tensor: cute.Tensor,
        tensor_smem_layout: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast: int
        ):
        tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(tensor_smem_layout, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, 
            gmem_tensor, 
            smem_layout, 
            smem_tile,
            mcast
        )
        return tma_atom, tma_tensor
    
    
    @staticmethod
    def _make_tma_store_atom_and_tensor(
        gmem_tensor: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int]
    ):
        tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )
        smem_tile = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op,
            gmem_tensor,
            smem_tile,
            epi_tile,
        )
        return tma_atom, tma_tensor
    
    
    @staticmethod
    def _compute_grid(
        d: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor D."""
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gd = cute.zipped_divide(d, tiler=c_shape)
        num_ctas_mn = gd[(0, (None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1),
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid    
    
    
    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided."""
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
    
    
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        #max_active_clusters: int,
        stream: cuda.CUstream
        ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        
        self._setup_attributes()
        
        tma_atom_a, tma_tensor_a = self._make_tma_atom_and_tensor(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1]
        )
        
        tma_atom_b, tma_tensor_b = self._make_tma_atom_and_tensor(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0]
        )
        
        tma_atom_c, tma_tensor_c = self._make_tma_store_atom_and_tensor(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile
        )
        
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes
            ]
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            epi_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.epi_num_stages * 2
            ]
    
        self.shared_storage = SharedStorage
        
        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_among_m,
            self.max_active_clusters
        )
        
        self.kernel(
            tma_tensor_a,
            tma_tensor_b,
            tma_tensor_c,
            scale_a,
            scale_b,
            tma_atom_a,
            tma_atom_b,
            tma_atom_c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.tiled_mma,
            self.cluster_layout_mn,
            tile_sched_params
        ).launch(
            grid= grid,
            block= [self.threads_per_cta, 1, 1],
            cluster= (*self.cluster_shape_mn, 1),
            min_blocks_per_mp= 1,
            stream=stream
        )
        
    
    @cute.kernel
    def kernel(
        self,
        A_mk: cute.Tensor,
        B_nk: cute.Tensor,
        C_mn: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        smem_a_layout_staged: cute.ComposedLayout,
        smem_b_layout_staged: cute.ComposedLayout,
        smem_epi_layout_staged: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        cluster_layout_mn: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        if tidx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mn = cluster_layout_mn.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(
            cluster_layout_mn, cluster_coord_mn, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cluster_layout_mn, cluster_coord_mn, mode=0
        )
        
        smem_tile_a_layout = cute.slice_(smem_a_layout_staged, (None, None, 0))
        smem_tile_b_layout = cute.slice_(smem_b_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, smem_tile_a_layout
            ) + cute.size_in_bytes(self.b_dtype, smem_tile_b_layout)
        
        # Alloc and init num_stagaes full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        producer_full_pipe = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_wg_mma
                )
        consumer_empty_pipe = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=producer_full_pipe,
            consumer_group=consumer_empty_pipe,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cute.make_layout((1, *cluster_layout_mn.shape, 1)),
            defer_sync=True,
        )
        epi_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads  # all MMA-WG threads collectively arrive per iteration
        )
        epi_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1  # single store warp's elected thread releases
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_pipeline_array_ptr.data_ptr(),
            num_stages=self.epi_num_stages,
            producer_group=epi_producer_group,
            consumer_group=epi_consumer_group
        )
        
        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        
        sA = storage.sA.get_tensor(
            smem_a_layout_staged.outer, swizzle=smem_a_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            smem_b_layout_staged.outer, swizzle=smem_b_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            smem_epi_layout_staged.outer, swizzle=smem_epi_layout_staged.inner
        )

        
        gA_mk = cute.local_tile(
            A_mk,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None)             
        )
        gB_nk = cute.local_tile(
            B_nk,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None)
        )
        gC_mn = cute.local_tile(
            C_mn,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None)
        )
        
        a_cluster_layout = cute.make_layout(self.cluster_shape_mn[1])
        a_cta_crd = cluster_coord_mn[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cluster_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2)
        )
        
        b_cluster_layout = cute.make_layout(self.cluster_shape_mn[0])
        b_cta_crd = cluster_coord_mn[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cluster_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2)
        )
        
        wg_idx = cute.arch.make_warp_uniform(
            tidx//self.threads_per_wg
        )
        mma_wg_thread_layout = cute.make_layout(
            self.num_wg_mma, stride= self.threads_per_wg
        )
        thr_mma = tiled_mma.get_slice(
            mma_wg_thread_layout(wg_idx - self.num_wg_dma)
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mn)
        
        acc_shape = tCgC.shape[:3]
        partial_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        acc =  cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        
        k_tile_cnt = cute.size(gA_mk, mode=[3])
        
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        
        is_dma_wg = wg_idx < self.num_wg_dma
        if is_dma_wg:
            cute.arch.setmaxregister_decrease(self.registers_dma)
        
        # producer warpgroup
        if warp_idx == 0:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m,n = tile_coord_mnl[0], tile_coord_mnl[1]
                tAgA_local = tAgA[(None, m, None)]
                tBgB_local = tBgB[(None, n, None)]
                """
                for prefetch_tile in cutlass.range(self.num_stages, unroll=1, at_least_once=True):
                    cute.prefetch(
                        tma_atom_a, 
                        tAgA_local[(None, prefetch_tile)]
                    )
                    cute.prefetch(
                        tma_atom_b,
                        tBgB_local[(None, prefetch_tile)]
                    )"""
                
                mainloop_producer_state.reset_count()
                peek_empty_status = cutlass.Boolean(1)
                if mainloop_producer_state.count < k_tile_cnt:
                    peek_empty_status = mainloop_pipeline.producer_try_acquire(
                        mainloop_producer_state
                    )
                
                for k_tile in cutlass.range(k_tile_cnt, at_least_once=True, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state, peek_empty_status)
                    tAgA_k = tAgA_local[(None, mainloop_producer_state.count)]
                    tBgB_k = tBgB_local[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    # tma load
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask = a_mcast_mask
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask = b_mcast_mask
                    )
                    """
                    if k_tile < k_tile_cnt - self.num_stages:
                        next_k_tile = mainloop_producer_state.count + self.num_stages
                        
                        cute.prefetch(
                            tma_atom_a, 
                            tAgA_local[(None, next_k_tile)]
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_local[(None, next_k_tile)]
                        )"""
                    
                    mainloop_producer_state.advance()
                    peek_empty_status = cutlass.Boolean(1)
                    if mainloop_producer_state.count < k_tile_cnt:
                        peek_empty_status = mainloop_pipeline.producer_try_acquire(
                            mainloop_producer_state
                        )
                    
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                    
            mainloop_pipeline.producer_tail(mainloop_producer_state)
            
        
        if not is_dma_wg:
            cute.arch.setmaxregister_increase(self.registers_mma)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            
            prologue_mma_cnt = 1
            scale_val = scale_a[0] * scale_b[0]
            num_k_blocks = cute.size(tCrA, mode=[2])
            
            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )  
            
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(),
                    4,
                ),
                self.c_dtype
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )
            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_wg_dma * self.num_warps_per_wg * 32
            )
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(acc)
            rC_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rC_layout = cute.make_layout(rC_shape[:3])
            tRS_rC = cute.make_rmem_tensor(tRS_rC_layout.shape, self.acc_dtype)
            tRS_rC_out = cute.make_rmem_tensor(tRS_rC_layout.shape, self.c_dtype)
            size_tRS_rC = cute.size(tRS_rC)
            
            epi_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_num_stages
            )
            while work_tile.is_valid_tile:
                
                tile_coord_mnl = work_tile.tile_idx
                m,n = tile_coord_mnl[0], tile_coord_mnl[1]
                #gC_mn_slice = gC_mn[(None, None, m, n)]
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                acc.fill(0.0)

                # Start with ACCUMULATE=False so first GMMA zeros accum_temp
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                )
                mma_count = 0

                cute.nvgpu.warpgroup.fence()
            
                for k_tile in cutlass.range(prologue_mma_cnt, at_least_once=True, unroll_full=True):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block, mainloop_consumer_read_state.index
                        )
                        cute.gemm(
                            tiled_mma,
                            partial_acc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            partial_acc
                        )
                        tiled_mma.set(
                                cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                            )
                    
                    cute.nvgpu.warpgroup.commit_group()
                    mma_count += num_k_blocks
                    if mma_count == self.mma_promotion_interval:
                        cute.nvgpu.warpgroup.wait_group(0)
                        # Element-wise promotion: accumulators += accum_temp
                        for i in range(cute.size(acc)):
                            acc[i] = acc[i] + partial_acc[i]
                        mma_count = 0
                        # Signal WGMMA to zero accum_temp on next instruction
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                        )
                    mainloop_consumer_read_state.advance()
                    
                # Main loop
                peek_full_status = cutlass.Boolean(1)
                if mainloop_consumer_read_state.count < k_tile_cnt:
                    peek_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)
                for k_tile in cutlass.range(prologue_mma_cnt, k_tile_cnt, at_least_once=True, unroll=2):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_full_status)
                    
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block, mainloop_consumer_read_state.index
                        )
                        cute.gemm(
                            tiled_mma,
                            partial_acc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            partial_acc
                        )
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                    
                    mma_count += num_k_blocks
                    """
                    if mma_count == self.mma_promotion_interval:
                        # Wait for all outstanding WGMMA writes to accum_temp
                        # before reading it
                        cute.nvgpu.warpgroup.wait_group(0)
                        # Element-wise promotion: accumulators += accum_temp
                        for i in range(cute.size(acc)):
                            acc[i] = acc[i] + partial_acc[i]
                        mma_count = 0
                        # Signal WGMMA to zero accum_temp on next instruction
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                        )"""
                    if warp_idx % self.num_warps_per_wg == 0:
                        mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                        mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()
                    peek_full_status = cutlass.Boolean(1)
                    if mainloop_consumer_read_state.count < k_tile_cnt:
                        peek_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)
                
                cute.nvgpu.warpgroup.wait_group(0)
                if mma_count > 0:
                    for i in range(cute.size(acc)):
                        acc[i] = acc[i] + partial_acc[i]
                        
                # Release remaining pipeline stages from prologue
                # for persistant kernel
                if warp_idx % self.num_warps_per_wg == 0:
                    for k_tile in range(prologue_mma_cnt):
                        mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                        mainloop_consumer_release_state.advance()

                # Apply scale_a * scale_b to accumulators
                for i in cutlass.range(cute.size(acc), unroll_full=True, at_least_once=True):
                    acc[i] = acc[i] * scale_val
                
                epi_tile_shape = cute.shape_div(self.tile_shape_mn, self.epi_tile)
                num_epi_tiles = cute.size(epi_tile_shape)
                
                # TODO count stored tiles for persistant kernel
                for epi_idx in cutlass.range_constexpr(num_epi_tiles):
                    epi_pipeline.producer_acquire(epi_producer_state)
                    for epi_v in cutlass.range_constexpr(size_tRS_rC):
                        tRS_rC[epi_v] = tRS_rAcc[epi_idx*size_tRS_rC + epi_v]
                    
                    acc_vec = tRS_rC.load()
                    tRS_rC_out.store(acc_vec.to(self.c_dtype))
                                        
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC_out,
                        tRS_sC[(None, None, None, epi_producer_state.index)]
                    )
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epi_pipeline.producer_commit(epi_producer_state)    # cheap arrive on "full" for this slot
                    epi_producer_state.advance()
                
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                                
                                        
        if warp_idx == self.epi_store_warp_id:
            is_elected = cute.arch.lane_idx() == 0
            epi_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_num_stages
            )
            epi_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_num_stages
            )
            
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m,n = tile_coord_mnl[0], tile_coord_mnl[1]
                gC_mn_slice = gC_mn[(None, None, m, n)]
                
                tCgC_for_tma_partition = cute.zipped_divide(gC_mn_slice, self.epi_tile)
                bSG_sC, bSG_gC = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition
                )
                
                num_epi_tiles = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1))
                prologue_epi_cnt = self.epi_num_stages - 1
                
                for epi_idx in cutlass.range(prologue_epi_cnt, unroll_full=True, at_least_once=True):
                    epi_pipeline.consumer_wait(epi_consumer_state)
                    
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, epi_consumer_state.index)],
                        bSG_gC[(None, gmem_coord)],
                        # evict first
                        cache_policy = cutlass.Int64(0x12F0000000000000)
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(1)
                    
                    epi_consumer_state.advance()
                
                for epi_idx in cutlass.range(prologue_epi_cnt, num_epi_tiles, unroll_full=True, at_least_once=True):
                    epi_pipeline.consumer_wait(epi_consumer_state)
                    
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, epi_consumer_state.index)],
                        bSG_gC[(None, gmem_coord)],
                        # evict first
                        cache_policy = cutlass.Int64(0x12F0000000000000)
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(1)
                    
                    epi_consumer_state.advance()
                    if is_elected:
                        epi_pipeline.consumer_release(epi_release_state)
                        epi_release_state.advance()
                cute.arch.cp_async_bulk_wait_group(0)
                if is_elected:
                    for _ in range(prologue_epi_cnt):
                        epi_pipeline.consumer_release(epi_release_state)
                        epi_release_state.advance()
                
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                
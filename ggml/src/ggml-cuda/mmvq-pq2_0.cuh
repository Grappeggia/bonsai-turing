// PQ2_0 mat-vec for 3-8 columns (plain 2D), tuned on Ada, also taken on Hopper, where it also serves 1-2 columns.
//
// Why: the generic mmvq kernel loses most of its DRAM throughput from 3 columns on (RTX 4070, Bonsai 2 27B shapes:
// 44% of peak at 4 columns). Here the activations use GGML_CUDA_Q8_1_PQ2: per column the qs bytes come first,
// permuted inside each 16-element group (position k*4+m holds element m*4+k), then one half2 per 32-block
// with d and the raw int16 sum of q. With that, (code_word >> 2k) & 0x03030303 pairs the raw 2-bit codes 0..3
// with the right activation bytes, so there is no per-weight decode. The digit bias is one integer subtraction:
// sum(q*(c-1)) = sum(q*c) - sum(q). Each warp handles up to 4 rows and reuses the activation slice of a 32-block for all of them.
#pragma once

#include "common.cuh"
#include "unary.cuh"

#include <mutex>
#include <unordered_map>

#define PQ2_0_MC_ROWS  4
#define PQ2_0_MC_WARPS 4

// Optional SoA-aligned PQ2_0 weight cache (uint2 qs[nrows * stride_row_x * 4] + half d[nrows * stride_row_x]):
// Eliminates 34-byte unaligned 5x LDG.16 per chunk -> single coalesced LDG.64 + cached half scale!
struct pq2_0_soa_entry {
    uint2 * qs;
    half  * d;
    int     nrows;
    int     stride_row_x;
};

__global__ static void pq2_0_repack_to_soa_kernel(
        const block_pq2_0 * __restrict__ src, uint2 * __restrict__ dst_qs, half * __restrict__ dst_d,
        const int nrows, const int stride_row_x) {
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const block_pq2_0 * row_src = src + (int64_t) row * stride_row_x;
    uint2 * row_qs = dst_qs + (int64_t) row * stride_row_x * 4;
    half  * row_d  = dst_d  + (int64_t) row * stride_row_x;
    for (int b = threadIdx.x; b < stride_row_x; b += blockDim.x) {
        const block_pq2_0 * blk = row_src + b;
        row_d[b] = blk->d;
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            const uint16_t * qp = (const uint16_t *) (blk->qs + kq * 8);
            uint2 w;
            w.x = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
            w.y = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
            row_qs[b * 4 + kq] = w;
        }
    }
}

static bool pq2_0_align_cache_enabled() {
    static const bool en = []() {
        const char * v = getenv("GGML_CUDA_PQ2_ALIGN_CACHE");
        return v != nullptr && atoi(v) != 0;
    }();
    return en;
}

static pq2_0_soa_entry get_or_repack_pq2_soa(
        const void * vx, const int nrows, const int stride_row_x, cudaStream_t stream) {
    static std::mutex mu;
    static std::unordered_map<const void *, pq2_0_soa_entry> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(vx);
    if (it != cache.end() && it->second.nrows == nrows && it->second.stride_row_x == stride_row_x) {
        return it->second;
    }
    pq2_0_soa_entry e{};
    e.nrows = nrows;
    e.stride_row_x = stride_row_x;
    const size_t qs_bytes = (size_t) nrows * (size_t) stride_row_x * 4 * sizeof(uint2);
    const size_t d_bytes  = (size_t) nrows * (size_t) stride_row_x * sizeof(half);
    if (cudaMalloc(&e.qs, qs_bytes) == cudaSuccess) {
        if (cudaMalloc(&e.d, d_bytes) == cudaSuccess) {
            pq2_0_repack_to_soa_kernel<<<(unsigned) nrows, 128, 0, stream>>>(
                (const block_pq2_0 *) vx, e.qs, e.d, nrows, stride_row_x);
            cache[vx] = e;
            return e;
        } else {
            (void) cudaGetLastError();
            cudaFree(e.qs);
            e.qs = nullptr;
        }
    } else {
        (void) cudaGetLastError();
    }
    e.qs = nullptr;
    e.d  = nullptr;
    return e;
}

// All weight reads of the kernel below go through this one helper. For 32-element chunk c of a row it returns the
// 8 code bytes (as two words: elements 0-15 and 16-31) and the block scale. It is the only place that knows the
// 34-byte block layout [d: half][qs: 32 bytes]; a repack of the weights only has to replace it.
template <bool aligned_soa = false>
static __device__ __forceinline__ void pq2_0_mc_load_chunk(
        const void * GGML_CUDA_RESTRICT vx, const half * GGML_CUDA_RESTRICT vxd,
        const int64_t row, const int stride_row_x, const int c,
        uint32_t & w0, uint32_t & w1, float & d) {
    if constexpr (aligned_soa) {
        const uint2 w = ((const uint2 *) vx)[row * (stride_row_x * 4) + c];
        w0 = w.x;
        w1 = w.y;
        d  = __half2float(__ldg(vxd + row * stride_row_x + (c >> 2)));
    } else {
        const block_pq2_0 * blk = (const block_pq2_0 *) vx + row*stride_row_x + (c >> 2);
        const uint16_t    * qp  = (const uint16_t *) (blk->qs + (c & 3)*8); // blocks are only 2-byte aligned
        d  = __half2float(blk->d);
        w0 = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
        w1 = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
    }
}

// Launch geometry (all template parameters, so they can be retuned independently of the kernel body):
//   rows   - output rows per warp-group (activation slice of a chunk is reused for all of them)
//   nwarps - warps per CTA
//   unroll - chunks per lane whose loads are issued together before they are consumed (order-preserving)
//   splitk - warps cooperating on one row group. 1: each group is one warp and the K reduction order of every row is
//            the same as before (lane c, c+32, ... then a shuffle tree), so any (rows, nwarps) is bit-exact with
//            any other splitk = 1 setting. >1: the chunk loop is strided by 32*splitk, the per-warp partial sums are
//            added in warp order 0..splitk-1 through shared memory, which changes the float summation order.
// Optional fusion (has_fusion, the same epilogue as mul_mat_vec_q): x_bias is added to the result; with has_gate a
// second weight matrix (gate, same layout and strides) is dotted against the same activations, gate_bias is added to
// it and the GLU op combines the two. Without fusion the epilogue is a plain store: a runtime bias check there costs
// the 8-column verify kernel ~12% on H200.
template <int ncols, bool has_fusion, bool has_gate, int rows = PQ2_0_MC_ROWS, int nwarps = PQ2_0_MC_WARPS, int splitk = 1, int unroll = 1, bool aligned_soa = false>
__global__ void __launch_bounds__(nwarps * 32, 3)
mul_mat_vec_pq2_0_mc(const void * vx_, const half * vxd_, const void * vy_, float * dst_,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const float * x_bias, const void * vgate_, const half * vgated_, const float * gate_bias, const ggml_glu_op glu_op, const float glu_limit) {
    static_assert(unroll == 1 || !has_gate, "unroll is not implemented for the fused gate variant");
    static_assert(nwarps % splitk == 0, "splitk must divide nwarps");
    static_assert(splitk == 1 || !has_gate, "split-K is not implemented for the fused gate variant");
    // Plain pointer parameters, restrict-qualified locals: GGML_CUDA_RESTRICT differs between the host and device
    // passes, which breaks the template stub. Same pattern as mul_mat_vec_q.
    const void * GGML_CUDA_RESTRICT vx  = vx_;
    const void * GGML_CUDA_RESTRICT vy  = vy_;
    float      * GGML_CUDA_RESTRICT dst = dst_;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int part = splitk == 1 ? 0 : warp % splitk;
    const int row0 = splitk == 1 ? ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * rows
                                 : (blockIdx.x * (nwarps / splitk) + warp / splitk) * rows;
    if (splitk == 1 && row0 >= nrows_x) {
        return; // with split-K every warp must reach the barrier, inactive groups only skip the stores
    }
    // launched through ggml_cuda_kernel_launch, which opts into PDL on Hopper and newer: wait for the kernel
    // that wrote vy (the q8_1 activation quantization) before reading it
    ggml_cuda_pdl_sync();

    const int     nchunk    = ncols_x / QK8_1;
    const size_t  col_bytes = (size_t) stride_col_y * sizeof(block_q8_1);
    const size_t  ds_off    = (size_t) stride_col_y * QK8_1; // qs region of a column, padded length
    const char *  y         = (const char *) vy;

    __shared__ float red[splitk > 1 ? nwarps*rows*ncols : 1];
    float acc[rows][ncols] = {};
    [[maybe_unused]] float accg[rows][ncols] = {};
    [[maybe_unused]] const void * GGML_CUDA_RESTRICT vgate = vgate_;

    constexpr int cstep = WARP_SIZE*splitk;
    // Each lane walks chunks part*32+lane, +cstep, ... exactly as before; `unroll` only issues the loads of that many
    // consecutive chunks of the lane before consuming them in the same order, so the accumulation order is unchanged.
    for (int cb = part*WARP_SIZE + lane; cb < nchunk; cb += cstep*unroll) {
        uint32_t w0[unroll][rows], w1[unroll][rows];
        float    d2[unroll][rows];
        [[maybe_unused]] uint32_t g0[rows], g1[rows];
        [[maybe_unused]] float    e2[rows];
        int4  a0[unroll][ncols], a1[unroll][ncols];
        float d8[unroll][ncols];
        int   qsum[unroll][ncols];
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            const int c = cb + u*cstep;
            if (c < nchunk) {
#pragma unroll
                for (int r = 0; r < rows; ++r) {
                    const int row = min(row0 + r, nrows_x - 1); // clamped rows are computed but not stored
                    pq2_0_mc_load_chunk<aligned_soa>(vx, vxd_, row, stride_row_x, c, w0[u][r], w1[u][r], d2[u][r]);
                    if constexpr (has_gate) {
                        pq2_0_mc_load_chunk<aligned_soa>(vgate, vgated_, row, stride_row_x, c, g0[r], g1[r], e2[r]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const char * yc = y + j*col_bytes;
                    a0[u][j] = *(const int4 *) (yc + c*QK8_1);
                    a1[u][j] = *(const int4 *) (yc + c*QK8_1 + 16);
                    const half2 ds = *(const half2 *) (yc + ds_off + c*sizeof(half2));
                    d8[u][j]   = __low2float(ds);
                    qsum[u][j] = __half_as_short(__high2half(ds));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            if (cb + u*cstep < nchunk) {
#pragma unroll
                for (int r = 0; r < rows; ++r) {
                    int t[8];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        t[k]     = (int) ((w0[u][r] >> (2*k)) & 0x03030303u);
                        t[4 + k] = (int) ((w1[u][r] >> (2*k)) & 0x03030303u);
                    }
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        int sumi = 0;
                        sumi = ggml_cuda_dp4a(t[0], a0[u][j].x, sumi);
                        sumi = ggml_cuda_dp4a(t[1], a0[u][j].y, sumi);
                        sumi = ggml_cuda_dp4a(t[2], a0[u][j].z, sumi);
                        sumi = ggml_cuda_dp4a(t[3], a0[u][j].w, sumi);
                        sumi = ggml_cuda_dp4a(t[4], a1[u][j].x, sumi);
                        sumi = ggml_cuda_dp4a(t[5], a1[u][j].y, sumi);
                        sumi = ggml_cuda_dp4a(t[6], a1[u][j].z, sumi);
                        sumi = ggml_cuda_dp4a(t[7], a1[u][j].w, sumi);
                        acc[r][j] += d2[u][r] * d8[u][j] * (float) (sumi - qsum[u][j]);
                    }
                    if constexpr (has_gate) {
                        int tg[8];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            tg[k]     = (int) ((g0[r] >> (2*k)) & 0x03030303u);
                            tg[4 + k] = (int) ((g1[r] >> (2*k)) & 0x03030303u);
                        }
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            int sumi = 0;
                            sumi = ggml_cuda_dp4a(tg[0], a0[u][j].x, sumi);
                            sumi = ggml_cuda_dp4a(tg[1], a0[u][j].y, sumi);
                            sumi = ggml_cuda_dp4a(tg[2], a0[u][j].z, sumi);
                            sumi = ggml_cuda_dp4a(tg[3], a0[u][j].w, sumi);
                            sumi = ggml_cuda_dp4a(tg[4], a1[u][j].x, sumi);
                            sumi = ggml_cuda_dp4a(tg[5], a1[u][j].y, sumi);
                            sumi = ggml_cuda_dp4a(tg[6], a1[u][j].z, sumi);
                            sumi = ggml_cuda_dp4a(tg[7], a1[u][j].w, sumi);
                            accg[r][j] += e2[r] * d8[u][j] * (float) (sumi - qsum[u][j]);
                        }
                    }
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int r = 0; r < rows; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float v = warp_reduce_sum<WARP_SIZE>(acc[r][j]);
            [[maybe_unused]] float gv = 0.0f;
            if constexpr (has_gate) {
                gv = warp_reduce_sum<WARP_SIZE>(accg[r][j]);
            }
            if constexpr (splitk > 1) {
                if (lane == 0) {
                    red[(warp*rows + r)*ncols + j] = v;
                }
            }
            const int row = row0 + r;
            if (splitk == 1 && lane == 0 && row < nrows_x) {
                const int64_t o = (int64_t) j*stride_col_dst + row;
                if constexpr (has_fusion) {
                    if (x_bias) {
                        v += x_bias[o];
                    }
                }
                if constexpr (has_gate) {
                    if (gate_bias) {
                        gv += gate_bias[o];
                    }
                    switch (glu_op) {
                        case GGML_GLU_OP_SWIGLU:     v *= ggml_cuda_op_silu_single(gv);          break;
                        case GGML_GLU_OP_GEGLU:      v *= ggml_cuda_op_gelu_single(gv);          break;
                        case GGML_GLU_OP_SWIGLU_OAI: v  = ggml_cuda_op_swiglu_oai_single(gv, v); break;
                        case GGML_GLU_OP_SWIGLU_CLAMP: v = ggml_cuda_op_swiglu_clamp_single(gv, v, glu_limit); break;
                        default:                     v *= gv;                                    break;
                    }
                }
                dst[o] = v;
            }
        }
    }

    if constexpr (splitk > 1) {
        __syncthreads();
        if (part == 0 && lane == 0) {
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                const int row = row0 + r;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (row < nrows_x) {
                        float v = 0.0f;
#pragma unroll
                        for (int p = 0; p < splitk; ++p) {
                            v += red[((warp + p)*rows + r)*ncols + j];
                        }
                        const int64_t o = (int64_t) j*stride_col_dst + row;
                        if constexpr (has_fusion) {
                            if (x_bias) {
                                v += x_bias[o];
                            }
                        }
                        dst[o] = v;
                    }
                }
            }
        }
    }
}

template <int ncols, int rows = PQ2_0_MC_ROWS, int nwarps = PQ2_0_MC_WARPS, int splitk = 1, int unroll = 1>
static void mul_mat_vec_pq2_0_mc_launch_rows(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_device & fusion) {
    const int ngroups = (nrows_x + rows - 1) / rows;
    const int groups_per_cta = nwarps / splitk;
    const dim3 block_nums((ngroups + groups_per_cta - 1) / groups_per_cta, 1, 1);
    const dim3 block_dims(nwarps * WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool use_soa = pq2_0_align_cache_enabled();
    pq2_0_soa_entry x_soa{}, g_soa{};
    bool soa_ok = false;
    if (use_soa) {
        x_soa = get_or_repack_pq2_soa(vx, nrows_x, stride_row_x, stream);
        soa_ok = (x_soa.qs != nullptr);
        if (soa_ok && fusion.gate) {
            g_soa = get_or_repack_pq2_soa(fusion.gate, nrows_x, stride_row_x, stream);
            soa_ok = (g_soa.qs != nullptr);
        }
    }

    if constexpr (splitk == 1 && unroll == 1) {
        if (fusion.gate) {
            if (soa_ok) {
                ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, true, rows, nwarps, splitk, unroll, true>, lp,
                    x_soa.qs, x_soa.d, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    (const float *) fusion.x_bias, g_soa.qs, g_soa.d, (const float *) fusion.gate_bias, fusion.glu_op, fusion.glu_limit);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, true, rows, nwarps, splitk, unroll, false>, lp,
                    vx, nullptr, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    (const float *) fusion.x_bias, fusion.gate, nullptr, (const float *) fusion.gate_bias, fusion.glu_op, fusion.glu_limit);
            }
            return;
        }
    } else {
        GGML_ASSERT(!fusion.gate);
    }
    if (fusion.x_bias) {
        if (soa_ok) {
            ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, false, rows, nwarps, splitk, unroll, true>, lp,
                x_soa.qs, x_soa.d, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                (const float *) fusion.x_bias, nullptr, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, false, rows, nwarps, splitk, unroll, false>, lp,
                vx, nullptr, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                (const float *) fusion.x_bias, nullptr, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
        }
    } else {
        if (soa_ok) {
            ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, false, false, rows, nwarps, splitk, unroll, true>, lp,
                x_soa.qs, x_soa.d, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                nullptr, nullptr, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, false, false, rows, nwarps, splitk, unroll, false>, lp,
                vx, nullptr, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                nullptr, nullptr, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
        }
    }
}

template <int ncols>
static void mul_mat_vec_pq2_0_mc_launch(const void *                            vx,
                                        const void *                            vy,
                                        float *                                 dst,
                                        const int                               ncols_x,
                                        const int                               nrows_x,
                                        const int                               stride_row_x,
                                        const int                               stride_col_y,
                                        const int                               stride_col_dst,
                                        cudaStream_t                            stream,
                                        const ggml_cuda_mm_fusion_args_device & fusion) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    static const bool turing_tune = [] { const char * e = getenv("GGML_CUDA_PQ2_TURING_TUNE"); return e != nullptr && atoi(e) != 0; }();
    if constexpr (ncols == 1) {
        // Smaller Hopper/Turing projections need more CTAs; keep four-row reuse for the vocabulary head.
        static const bool splitk_env = [] { const char * e = getenv("GGML_CUDA_PQ2_SPLITK"); return e != nullptr && atoi(e) != 0; }();
        if ((cc == GGML_CUDA_CC_HOPPER || (cc == GGML_CUDA_CC_TURING && turing_tune)) && splitk_env && !fusion.gate &&
                ((ncols_x == 12288 && nrows_x == 4096) || (ncols_x == 17408 && nrows_x == 5120))) {
            mul_mat_vec_pq2_0_mc_launch_rows<ncols, 2, 8, 2>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y,
                                                             stride_col_dst, stream, fusion);
            return;
        }
        if ((cc == GGML_CUDA_CC_HOPPER || (cc == GGML_CUDA_CC_TURING && turing_tune)) && nrows_x <= 17408) {
            mul_mat_vec_pq2_0_mc_launch_rows<ncols, 2>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y,
                                                       stride_col_dst, stream, fusion);
            return;
        }
    }
    mul_mat_vec_pq2_0_mc_launch_rows<ncols>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                                            stream, fusion);
}

static void mul_mat_vec_pq2_0_mc_switch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_device & fusion) {
    GGML_ASSERT(ncols_x % QK_PQ2_0 == 0);
    switch (ncols_dst) {
        case 1: mul_mat_vec_pq2_0_mc_launch<1>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 2: mul_mat_vec_pq2_0_mc_launch<2>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 3: mul_mat_vec_pq2_0_mc_launch<3>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 4: mul_mat_vec_pq2_0_mc_launch<4>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 5: mul_mat_vec_pq2_0_mc_launch<5>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 6: mul_mat_vec_pq2_0_mc_launch<6>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 7: mul_mat_vec_pq2_0_mc_launch<7>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 8: mul_mat_vec_pq2_0_mc_launch<8>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        default: GGML_ABORT("unsupported column count for the PQ2_0 multi-column kernel");
    }
}

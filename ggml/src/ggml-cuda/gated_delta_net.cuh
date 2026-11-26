#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// The fused recurrent-state gather (ggml_cuda_gated_delta_net_gather, common.cuh) is looked up in
// ctx.gdn_gathers() by node pointer; the graph evaluator registers it when it skips the GET_ROWS.
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// fused-kernel replay output: base -> plane 0, records -> bank rec_rows[seq] (when rec != nullptr)
struct ggml_cuda_gdn_replay_fused {
    static constexpr int MAX_BATCH_SEQS = 64;
    float *         base                  = nullptr;
    float *         rec                   = nullptr;
    const int64_t * rec_rows              = nullptr;
    int64_t         rec_row_stride        = 0;
    // Batched post-replay SET_ROWS for n_seqs > 1 (100% cell-swap safe):
    float *         rec_batch_dst         = nullptr;
    const int64_t * rec_batch_wrows[MAX_BATCH_SEQS] = {};
    int             rec_batch_n_seqs      = 0;
    int64_t         rec_batch_cell_stride = 0;
    int64_t         rec_batch_bank_stride = 0;
};

void ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                         const ggml_cuda_gdn_replay_fused * fused = nullptr);

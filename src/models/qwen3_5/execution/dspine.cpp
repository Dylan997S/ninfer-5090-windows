#include "models/qwen3_5/execution/dspine.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/dspine.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5::execution {
void dspine_write(DraftExecutionView& execution, Tensor& residual, const Tensor& anchors, int layer,
                  int width, workspace::DSpineRoots& saved) {
    const auto& bridge = *execution.parameters.draft->dspine;
    auto scope         = execution.work.scope();
    auto roots         = workspace::dspine_bridge(execution.work, residual.ne[1]);
    ops::dspine_prepare_read(residual, roots.normalized, saved.rms, width, 1e-6f, execution.cuda);
    project(roots.normalized, bridge.reads.at(layer), roots.codes, execution.work, execution.cuda);
    ops::dspine_normalize_codes(roots.codes, bridge.codebook, anchors, width, 1e-6f,
                                execution.cuda);
    project(roots.codes, bridge.projection, saved.packed, execution.work, execution.cuda);
    ops::dspine_message(saved.packed, bridge.gate_bias, width, roots.message, execution.cuda);
    project(roots.message, bridge.writes.at(layer), roots.delta, execution.work, execution.cuda);
    if (layer + 1 == static_cast<int>(bridge.reads.size()))
        CUDA_CHECK(cudaMemcpyAsync(saved.raw.data, residual.data, residual.bytes(),
                                   cudaMemcpyDeviceToDevice, execution.cuda.stream()));
    ops::dspine_finish(roots.delta, saved.rms, width, residual, execution.cuda);
}

void dspine_proposals(DraftExecutionView& execution, const Tensor& residual, const Tensor& anchors,
                      int width, workspace::DSpineRoots& saved, Tensor& drafts) {
    const auto& parameters = *execution.parameters.draft;
    const auto& bridge     = *parameters.dspine;
    const int batch        = residual.ne[1] / width;
    const int rows         = (width - 1) * batch;
    auto& work             = execution.work;
    auto hidden            = work.alloc(DType::BF16, {5120, rows});
    {
        auto scope = work.scope();
        auto normalized = work.alloc(DType::BF16, {5120, width * batch});
        ops::dspine_post_write_norm(residual, parameters.final_norm, 1e-6f, normalized,
                                     execution.cuda);
        const std::size_t row_bytes = std::size_t(5120) * (width - 1) * sizeof(std::uint16_t);
        const std::size_t source_pitch = std::size_t(5120) * width * sizeof(std::uint16_t);
        const auto* source = static_cast<const std::byte*>(normalized.data) +
                             std::size_t(5120) * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy2DAsync(hidden.data, row_bytes, source, source_pitch, row_bytes,
                                     batch, cudaMemcpyDeviceToDevice, execution.cuda.stream()));
    }
    auto candidates = work.alloc(DType::I32, {16, rows});
    {
        auto scope  = work.scope();
        auto scores = work.alloc(DType::FP32, {16, rows});
        ops::linear_topk(hidden, parameters.output_head.weight,
                         dimension(execution.parameters.profile->public_tokens),
                         candidates, scores, work, execution.cuda);
    }
    if (bridge.sender_table && bridge.head_write_table) {
        auto base_scores = work.alloc(DType::FP32, {16, rows});
        auto scores = work.alloc(DType::FP32, {256, rows});
        ops::dspine_table_greedy_path(
            saved.raw, saved.rms, saved.packed, bridge.gate_bias, *bridge.sender_table,
            *bridge.head_write_table, parameters.final_norm, parameters.output_head.weight,
            candidates, anchors, width, base_scores, scores, drafts, execution.cuda);
        return;
    }
    auto delta = work.alloc(DType::BF16, {5120, 16 * rows});
    {
        auto scope = work.scope();
        auto roots = workspace::dspine_refine(work, 16 * rows);
        ops::dspine_sender_codes(bridge.codebook, candidates, anchors, width, roots.codes,
                                 execution.cuda);
        project(roots.codes, bridge.projection, roots.packed, work, execution.cuda);
        ops::dspine_refine_message(saved.packed, roots.packed, bridge.gate_bias, width,
                                   roots.message, execution.cuda);
        project(roots.message, bridge.writes.back(), delta, work, execution.cuda);
    }
    auto scores = work.alloc(DType::FP32, {256, rows});
    ops::dspine_greedy_path(saved.raw, saved.rms, delta, parameters.final_norm,
                            parameters.output_head.weight, candidates, width, scores, drafts,
                            execution.cuda);
}
} // namespace ninfer::models::qwen3_5::execution

#include "ops/common/host/operands.h"
#include "ninfer/ops/candidate_selector.h"

#include "ops/sampling/candidate_path/plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;
constexpr const char* kOp            = "candidate_selector_path";

void require_codebook(const Weight& codebook, const char* label) {
    if (codebook.qtype == QType::NVFP4) {
        if (codebook.n != kCodebookRows || codebook.k != kRank) {
            throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
        }
        (void)common::validate_nvfp4_weight(codebook, kOp);
        return;
    }
    constexpr std::uint64_t kPayloadBytes =
        static_cast<std::uint64_t>(kCodebookRows) * kRank * sizeof(std::uint16_t);
    if (codebook.qtype != QType::BF16 || codebook.layout != QuantLayout::Contiguous ||
        codebook.ndim != 2 || codebook.n != kCodebookRows || codebook.k != kRank ||
        codebook.shape[0] != kCodebookRows || codebook.shape[1] != kRank ||
        codebook.padded_shape[0] != kCodebookRows || codebook.padded_shape[1] != kRank ||
        codebook.payload_bytes < kPayloadBytes || codebook.qhigh != nullptr ||
        codebook.high_plane_bytes != 0 ||
        !common::operand::aligned(codebook.qdata, 16)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
    }
}

} // namespace

std::size_t candidate_selector_path_workspace_capacity_bytes(int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 32 || max_batch < min_batch)
        throw std::invalid_argument("selector workspace: invalid K/B interval");
    WorkspaceLayoutBuilder layout;
    for (int k = min_steps; k <= max_steps; ++k)
        for (int b = min_batch; b <= max_batch; ++b) {
            auto scope = layout.scope();
            (void)candidate_path_detail::allocate_selector_workspace(
                layout, candidate_path_detail::select_candidate_path(k, b), k, b);
        }
    return layout.peak_bytes();
}

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Weight& predecessor_codebook, const Weight& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, MemoryArena& workspace,
                             CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("candidate_selector_path: workspace belongs to another device");
    }
    const auto stream             = execution.stream();
    const std::int32_t batch_size = candidate_ids.ne[2];
    const std::int32_t kSteps     = candidate_ids.ne[1];
    if (kSteps < 1 || kSteps > 15)
        throw std::invalid_argument("candidate_selector_path: K must be in [1,15]");
    if (batch_size < 1 || batch_size > 32) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,32]");
    }
    common::operand::tensor(candidate_ids, DType::I32, {kCandidates, kSteps, batch_size, 1}, 16,
                            kOp, "candidate_ids", execution.device());
    common::operand::tensor(unary_scores, DType::FP32, {kCandidates, kSteps, batch_size, 1}, 16, kOp,
                            "unary_scores", execution.device());
    common::operand::tensor(projected_hidden, DType::BF16, {kRank, kSteps, batch_size, 1}, 16, kOp,
                            "projected_hidden", execution.device());
    common::operand::tensor(anchors, DType::I32, {batch_size, 1, 1, 1}, 16, kOp, "anchors",
                            execution.device());
    require_codebook(predecessor_codebook, "predecessor_codebook");
    require_codebook(successor_codebook, "successor_codebook");
    // Each codebook is validated on its own, and the route below is chosen from the predecessor
    // alone, so a mixed pair passes validation and then reaches one kernel holding the other
    // format's payload. Measured on sm_120a: an NVFP4 predecessor with a BF16 successor makes the
    // NVFP4 kernel dereference the successor's null scales pointer, an illegal access 0x38AC into
    // unmapped memory, which aborts the process rather than failing the request. The reverse reads
    // packed E2M1 as bf16 and returns wrong drafts. Both codebooks come from one artifact in
    // production, so requiring them to agree closes the hole at no cost.
    if (predecessor_codebook.qtype != successor_codebook.qtype) {
        throw std::invalid_argument("candidate_selector_path: codebook formats must match");
    }
    common::operand::tensor(base_positions, DType::I32, {batch_size, 1, 1, 1}, 16, kOp,
                            "base_positions", execution.device());
    common::operand::tensor(drafts, DType::I32, {kSteps, batch_size, 1, 1}, 16, kOp, "drafts",
                            execution.device());
    common::operand::tensor(proposal_q, DType::FP32, {kCandidates, kSteps, batch_size, 1}, 16, kOp,
                            "proposal_q", execution.device());
    if (!common::operand::aligned(configs, alignof(SamplingConfig))) {
        throw std::invalid_argument("candidate_selector_path: invalid configs");
    }
    const std::array ranges{
        common::operand::range(candidate_ids, kOp),
        common::operand::range(unary_scores, kOp),
        common::operand::range(projected_hidden, kOp),
        common::operand::range(anchors, kOp),
        common::operand::range(predecessor_codebook.qdata, predecessor_codebook.payload_bytes, kOp),
        common::operand::range(successor_codebook.qdata, successor_codebook.payload_bytes, kOp),
        common::operand::range(base_positions, kOp),
        common::operand::range(drafts, kOp),
        common::operand::range(proposal_q, kOp),
        common::operand::range(configs, batch_size * sizeof(SamplingConfig), kOp)};
    common::operand::disjoint(ranges, kOp);

    candidate_path_detail::execute_candidate_path(candidate_ids, unary_scores, projected_hidden,
                                                  anchors, predecessor_codebook,
                                                  successor_codebook, base_positions, configs,
                                                  drafts, proposal_q, workspace, stream);
}

} // namespace ninfer::ops

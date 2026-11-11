#include "ops/common/host/operands.h"
#include "ops/sampling/candidate_path/plan.h"
#include "ops/sampling/candidate_path/launch.h"
#include <stdexcept>

namespace ninfer::ops::candidate_path_detail {
SelectorRoute select_candidate_path(int steps, int batch) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 32)
        throw std::invalid_argument("invalid selector K/B");
    return steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice;
}

void execute_candidate_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                            const Tensor& projected_hidden, const Tensor& anchors,
                            const Weight& predecessor_codebook, const Weight& successor_codebook,
                            const Tensor& base_positions, const SamplingConfig* configs,
                            Tensor& drafts, Tensor& proposal_q, MemoryArena& workspace,
                            cudaStream_t stream) {
    const auto route = select_candidate_path(candidate_ids.ne[1], candidate_ids.ne[2]);
    // One allocation for the whole call, whichever route runs: the NVFP4 route taking its own
    // from the same arena made a Lattice call need twice the queried capacity.
    auto scope         = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    constexpr const char* kOp = "candidate_selector_path";
    const Tensor* live[]{
        &candidate_ids, &unary_scores, &projected_hidden, &anchors,
        &base_positions, &drafts,       &proposal_q};
    for (const auto* work : {&scratch.edges}) {
        if (!work->data) continue;
        const auto range = common::operand::range(*work, "candidate_selector_path workspace");
        for (const auto* tensor : live)
            common::operand::disjoint(range, common::operand::range(*tensor, kOp),
                                      "candidate_selector_path workspace");
        // The codebooks are weights, not tensors: a packed NVFP4 parent has no BF16 tensor form,
        // so its packed payload is what the workspace has to stay clear of.
        for (const auto* codebook : {&predecessor_codebook, &successor_codebook})
            common::operand::disjoint(
                range, common::operand::range(codebook->qdata, codebook->payload_bytes, kOp),
                "candidate_selector_path workspace");
        common::operand::disjoint(
            range,
            common::operand::range(configs, candidate_ids.ne[2] * sizeof(SamplingConfig), kOp),
            "candidate_selector_path workspace");
    }
    if (predecessor_codebook.qtype == QType::NVFP4) {
        execute_candidate_path_nvfp4(candidate_ids, unary_scores, projected_hidden, anchors,
                                     predecessor_codebook, successor_codebook, base_positions,
                                     configs, drafts, proposal_q, scratch, stream);
        return;
    }
    launch_candidate_path(route, candidate_ids, unary_scores, projected_hidden, anchors,
                          predecessor_codebook, successor_codebook, base_positions, configs, drafts,
                          proposal_q, scratch, stream);
}
} // namespace ninfer::ops::candidate_path_detail

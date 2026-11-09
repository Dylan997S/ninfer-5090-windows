#pragma once
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/workspace.h"

namespace ninfer::models::qwen3_5::execution {
void dspine_write(DraftExecutionView&, Tensor&, const Tensor&, int, int, workspace::DSpineRoots&);
void dspine_proposals(DraftExecutionView&, const Tensor&, const Tensor&, int,
                      workspace::DSpineRoots&, Tensor&);
} // namespace ninfer::models::qwen3_5::execution

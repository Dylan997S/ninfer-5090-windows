#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

namespace ninfer::ops {
struct SamplingConfig;

// DSpine's represented BF16 code-bridge operations. Matrices are contiguous [features, rows].
// All operands live on the execution device. Scratch and outputs belong to the caller.
// Read normalization is x / (sqrt(mean(x*x)) + eps), distinct from RMSNorm.
void dspine_prepare_read(const Tensor& raw, Tensor& normalized, Tensor& rms, int width, float eps,
                         CudaExecutionView execution);
// Normalize predicted codes to length sqrt(C), replacing each anchor with sqrt(C)*codebook[id].
void dspine_normalize_codes(Tensor& codes, const Tensor& codebook, const Tensor& anchors, int width,
                            float eps, CudaExecutionView execution);
// Packed rows contain [receiver, sender, value]. No edge connects adjacent requests.
void dspine_message(const Tensor& packed, const Tensor& bias, int width, Tensor& message,
                    CudaExecutionView execution);
void dspine_finish(const Tensor& delta, const Tensor& rms, int width, Tensor& residual,
                   CudaExecutionView execution);
// Gather exact predecessor codes from the preceding candidate row, or the request anchor.
void dspine_sender_codes(const Tensor& codebook, const Tensor& candidates, const Tensor& anchors,
                         int width, Tensor& codes, CudaExecutionView execution);
void dspine_refine_message(const Tensor& receiver, const Tensor& senders, const Tensor& bias,
                           int width, Tensor& message, CudaExecutionView execution);
// Re-score every candidate against every exact predecessor using the last write's residual.
// With null configs, omit the common RMS denominator. With device configs[batch],
// normalize logits before applying presence/frequency penalties to committed counts
// plus the already selected draft prefix. Counts include the anchor. Other sampling
// fields are intentionally ignored: these are deterministic point-mass proposals.
// scores is FP32 [16,16,(width-1)*batch], drafts is I32 [width-1,batch]. The deterministic
// proposal is verified as a point mass by the shared target acceptance implementation.
void dspine_greedy_path(const Tensor& raw, const Tensor& rms, const Tensor& delta,
                        const Tensor& norm, const Weight& head, const Tensor& candidates, int width,
                        Tensor& scores, Tensor& drafts, CudaExecutionView execution,
                        const SamplingConfig* configs = nullptr);

// Official low-rank table form. All IDs must be in [0, head.n). Tables are precomputed
// BF16 [2*512, vocabulary] senders and [512, vocabulary] head/write products. The
// FP32 message remains unrounded, as in the reference serial-table implementation.
// Caller owns scratch base_scores [16,positions] and scores [256,positions].
void dspine_table_greedy_path(const Tensor& raw, const Tensor& rms, const Tensor& receiver,
                              const Tensor& bias, const Tensor& sender_table,
                              const Tensor& score_table, const Tensor& norm, const Weight& head,
                              const Tensor& candidates, const Tensor& anchors, int width,
                              Tensor& base_scores, Tensor& scores, Tensor& drafts,
                              CudaExecutionView execution);

// Post-write RMSNorm follows the checkpoint's two BF16 rounding points, first
// after normalization and then after multiplication by the represented norm.
// Exactly in-place output is supported. Partial overlaps are rejected.
void dspine_post_write_norm(const Tensor& raw, const Tensor& norm, float eps, Tensor& output,
                            CudaExecutionView execution);

} // namespace ninfer::ops

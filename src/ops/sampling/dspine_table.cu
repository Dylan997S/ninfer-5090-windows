#include "ninfer/ops/dspine.h"
#include "ops/common/host/operands.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace ninfer::ops {
namespace {
using Bf16                = __nv_bfloat16;
constexpr int Hidden      = 5120;
constexpr int Rank        = 512;
constexpr int Candidates  = 16;
constexpr int BaseThreads = 256;

__global__ void base_kernel(const Bf16* raw, const Bf16* norm, const signed char* head,
                            const __half* scales, const int* candidates, int width, float* base) {
    __shared__ float sums[BaseThreads];
    const int candidate = blockIdx.x;
    const int position  = blockIdx.y;
    const int original  = position / (width - 1) * width + position % (width - 1) + 1;
    const int token     = candidates[position * Candidates + candidate];
    float value         = 0;
    for (int h = threadIdx.x; h < Hidden; h += BaseThreads) {
        const long long offset = static_cast<long long>(token) * Hidden + h;
        const float weight     = __bfloat162float(
            __float2bfloat16(float(head[offset]) * __half2float(scales[offset / 32])));
        value += weight * __bfloat162float(norm[h]) * __bfloat162float(raw[original * Hidden + h]);
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = BaseThreads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) base[position * Candidates + candidate] = sums[0];
}

__global__ void table_kernel(const float* base, const float* rms, const Bf16* receiver,
                             const Bf16* bias, const Bf16* sender_table, const Bf16* score_table,
                             const int* candidates, const int* anchors, int width, float* scores) {
    __shared__ float message[Rank];
    const int predecessor         = blockIdx.x;
    const int position            = blockIdx.y;
    const int step                = position % (width - 1);
    const int original            = position / (width - 1) * width + step + 1;
    const int sender              = step ? candidates[(position - 1) * Candidates + predecessor]
                                         : anchors[position / (width - 1)];
    const int r                   = threadIdx.x;
    const long long sender_offset = static_cast<long long>(sender) * (2 * Rank);
    const float query             = __bfloat162float(receiver[original * 3 * Rank + r]);
    const float gate              = __bfloat162float(sender_table[sender_offset + r]);
    const float value             = __bfloat162float(sender_table[sender_offset + Rank + r]);
    message[r]                    = value / (1 + expf(-(query + gate + __bfloat162float(bias[r]))));
    __syncthreads();

    const int candidate = threadIdx.x / 32;
    const int lane      = threadIdx.x % 32;
    const int token     = candidates[position * Candidates + candidate];
    float dot           = 0;
    for (int column = lane; column < Rank; column += 32)
        dot += message[column] *
               __bfloat162float(score_table[static_cast<long long>(token) * Rank + column]);
    for (int offset = 16; offset; offset /= 2) dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if (!lane)
        scores[(position * Candidates + predecessor) * Candidates + candidate] =
            base[position * Candidates + candidate] + rms[original] * dot;
}

__global__ void chain_kernel(const float* scores, const int* candidates, int width, int* drafts) {
    const int batch = blockIdx.x;
    int predecessor = 0;
    for (int step = 0; step < width - 1; ++step) {
        const int position = batch * (width - 1) + step;
        const float* row   = scores + (position * Candidates + predecessor) * Candidates;
        int best           = 0;
        for (int candidate = 1; candidate < Candidates; ++candidate)
            if (row[candidate] > row[best] ||
                (row[candidate] == row[best] && candidates[position * Candidates + candidate] <
                                                    candidates[position * Candidates + best]))
                best = candidate;
        drafts[position] = candidates[position * Candidates + best];
        predecessor      = best;
    }
}
} // namespace

void dspine_table_greedy_path(const Tensor& raw, const Tensor& rms, const Tensor& receiver,
                              const Tensor& bias, const Tensor& sender_table,
                              const Tensor& score_table, const Tensor& norm, const Weight& head,
                              const Tensor& candidates, const Tensor& anchors, int width,
                              Tensor& base_scores, Tensor& scores, Tensor& drafts,
                              CudaExecutionView execution) {
    constexpr const char* op = "DSpine table";
    const int rows           = raw.ne[1];
    if (width < 2 || width > 16 || rows < width || rows > width * 32 || rows % width)
        throw std::invalid_argument("DSpine table requires 2..16-token blocks and 1..32 requests");
    const int batch     = rows / width;
    const int positions = batch * (width - 1);
    const auto device   = execution.device();
    const int vocab     = sender_table.ne[1];
    if (vocab < Candidates || score_table.ne[1] != vocab)
        throw std::invalid_argument("DSpine table vocabulary mismatch");
    common::operand::tensor(raw, DType::BF16, {Hidden, rows}, 2, op, "raw", device);
    common::operand::tensor(rms, DType::FP32, {rows}, 4, op, "rms", device);
    common::operand::tensor(receiver, DType::BF16, {3 * Rank, rows}, 2, op, "receiver", device);
    common::operand::tensor(bias, DType::BF16, {Rank}, 2, op, "bias", device);
    common::operand::tensor(sender_table, DType::BF16, {2 * Rank, vocab}, 2, op, "senders", device);
    common::operand::tensor(score_table, DType::BF16, {Rank, vocab}, 2, op, "scores table", device);
    common::operand::tensor(norm, DType::BF16, {Hidden}, 2, op, "norm", device);
    common::operand::tensor(candidates, DType::I32, {Candidates, positions}, 4, op, "candidates",
                            device);
    common::operand::tensor(anchors, DType::I32, {batch}, 4, op, "anchors", device);
    common::operand::tensor(base_scores, DType::FP32, {Candidates, positions}, 4, op, "base",
                            device);
    common::operand::tensor(scores, DType::FP32, {Candidates * Candidates, positions}, 4, op,
                            "scores", device);
    common::operand::tensor(drafts, DType::I32, {width - 1, batch}, 4, op, "drafts", device);
    if (head.device != device || head.qtype != QType::Q8_G32_FP16 || head.k != Hidden ||
        head.n != vocab || head.group != 32 || head.layout != QuantLayout::RowSplit ||
        !head.qdata || !head.scales)
        throw std::invalid_argument("DSpine table requires matching Q8 group-32 head");
    const std::array outputs{common::operand::range(base_scores, op),
                             common::operand::range(scores, op),
                             common::operand::range(drafts, op)};
    common::operand::disjoint(outputs, op);
    for (const Tensor* input :
         {&raw, &rms, &receiver, &bias, &sender_table, &score_table, &norm, &candidates, &anchors})
        for (const auto output : outputs)
            common::operand::disjoint(common::operand::range(*input, op), output, op);
    base_kernel<<<dim3(Candidates, positions), BaseThreads, 0, execution.stream()>>>(
        static_cast<const Bf16*>(raw.data), static_cast<const Bf16*>(norm.data),
        static_cast<const signed char*>(head.qdata), static_cast<const __half*>(head.scales),
        static_cast<const int*>(candidates.data), width, static_cast<float*>(base_scores.data));
    CUDA_CHECK(cudaGetLastError());
    table_kernel<<<dim3(Candidates, positions), Rank, 0, execution.stream()>>>(
        static_cast<const float*>(base_scores.data), static_cast<const float*>(rms.data),
        static_cast<const Bf16*>(receiver.data), static_cast<const Bf16*>(bias.data),
        static_cast<const Bf16*>(sender_table.data), static_cast<const Bf16*>(score_table.data),
        static_cast<const int*>(candidates.data), static_cast<const int*>(anchors.data), width,
        static_cast<float*>(scores.data));
    CUDA_CHECK(cudaGetLastError());
    chain_kernel<<<batch, 1, 0, execution.stream()>>>(static_cast<const float*>(scores.data),
                                                      static_cast<const int*>(candidates.data),
                                                      width, static_cast<int*>(drafts.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops

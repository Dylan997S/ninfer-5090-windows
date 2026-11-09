#include "ninfer/ops/dspine.h"
#include "ninfer/ops/sampling.h"
#include "ops/common/host/operands.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
using Bf16               = __nv_bfloat16;
constexpr int Threads    = 256;
constexpr int Code       = 1024;
constexpr int Rank       = 512;
constexpr int Hidden     = 5120;
constexpr int Candidates = 16;

__device__ float sum_block(float value) {
    __shared__ float sums[Threads];
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    return sums[0];
}

__global__ void read_kernel(const Bf16* raw, Bf16* output, float* rms, float eps) {
    const int row = blockIdx.x;
    float squared = 0;
    for (int h = threadIdx.x; h < Hidden; h += Threads) {
        const float x = __bfloat162float(raw[row * Hidden + h]);
        squared += x * x;
    }
    const float scale = sqrtf(fmaxf(sum_block(squared) / Hidden, 1.1754943508222875e-38f));
    if (!threadIdx.x) rms[row] = scale;
    for (int h = threadIdx.x; h < Hidden; h += Threads)
        output[row * Hidden + h] =
            __float2bfloat16(__bfloat162float(raw[row * Hidden + h]) / (scale + eps));
}

__global__ void normalize_kernel(Bf16* codes, const Bf16* table, const int* anchors, int width,
                                 float eps) {
    const int row = blockIdx.x;
    float squared = 0;
    for (int c = threadIdx.x; c < Code; c += Threads) {
        const float x = __bfloat162float(codes[row * Code + c]);
        squared += x * x;
    }
    const float length = sqrtf(sum_block(squared));
    for (int c = threadIdx.x; c < Code; c += Threads) {
        const float x =
            row % width == 0
                ? __bfloat162float(table[static_cast<long long>(anchors[row / width]) * Code + c])
                : __bfloat162float(codes[row * Code + c]) / (length + eps);
        codes[row * Code + c] = __float2bfloat16(sqrtf(float(Code)) * x);
    }
}

__global__ void message_kernel(const Bf16* packed, const Bf16* bias, int width, Bf16* output) {
    const int row = blockIdx.x;
    for (int r = threadIdx.x; r < Rank; r += Threads) {
        float value = 0;
        if (row % width) {
            const float q = __bfloat162float(packed[row * 3 * Rank + r]);
            const float s = __bfloat162float(packed[(row - 1) * 3 * Rank + Rank + r]);
            value         = __bfloat162float(packed[(row - 1) * 3 * Rank + 2 * Rank + r]) /
                            (1 + expf(-(q + s + __bfloat162float(bias[r]))));
        }
        output[row * Rank + r] = __float2bfloat16(value);
    }
}

__global__ void finish_kernel(const Bf16* delta, const float* rms, int width, Bf16* residual) {
    const int row = blockIdx.x;
    if (row % width == 0) return;
    for (int h = threadIdx.x; h < Hidden; h += Threads) {
        const int i = row * Hidden + h;
        residual[i] =
            __float2bfloat16(__bfloat162float(residual[i]) + rms[row] * __bfloat162float(delta[i]));
    }
}

__global__ void sender_kernel(const Bf16* table, const int* candidates, const int* anchors,
                              int width, Bf16* output) {
    const int row      = blockIdx.x;
    const int position = row / Candidates;
    const int step     = position % (width - 1);
    const int token    = step == 0 ? anchors[position / (width - 1)]
                                   : candidates[(position - 1) * Candidates + row % Candidates];
    for (int c = threadIdx.x; c < Code; c += Threads)
        output[row * Code + c] = __float2bfloat16(
            32.f * __bfloat162float(table[static_cast<long long>(token) * Code + c]));
}

__global__ void refine_kernel(const Bf16* receiver, const Bf16* sender, const Bf16* bias, int width,
                              Bf16* message) {
    const int row      = blockIdx.x;
    const int position = row / Candidates;
    const int original = position / (width - 1) * width + position % (width - 1) + 1;
    for (int r = threadIdx.x; r < Rank; r += Threads) {
        const float q = __bfloat162float(receiver[original * 3 * Rank + r]);
        const float s = __bfloat162float(sender[row * 3 * Rank + Rank + r]);
        const float v = __bfloat162float(sender[row * 3 * Rank + 2 * Rank + r]);
        message[row * Rank + r] =
            __float2bfloat16(v / (1 + expf(-(q + s + __bfloat162float(bias[r])))));
    }
}

__global__ void score_kernel(const Bf16* raw, const float* rms, const Bf16* delta, const Bf16* norm,
                             const signed char* head, const __half* scales, const int* candidates,
                             int width, float* scores, bool normalize) {
    const int candidate   = blockIdx.x;
    const int predecessor = blockIdx.y;
    const int position    = blockIdx.z;
    const int original    = position / (width - 1) * width + position % (width - 1) + 1;
    const int token       = candidates[position * Candidates + candidate];
    float value           = 0;
    float squared         = 0;
    for (int h = threadIdx.x; h < Hidden; h += Threads) {
        const long long offset = static_cast<long long>(token) * Hidden + h;
        const float weight     = __bfloat162float(
            __float2bfloat16(float(head[offset]) * __half2float(scales[offset / 32])));
        const float residual =
            __bfloat162float(raw[original * Hidden + h]) +
            rms[original] *
                __bfloat162float(delta[(position * Candidates + predecessor) * Hidden + h]);
        value += weight * __bfloat162float(norm[h]) * residual;
        if (normalize) squared += residual * residual;
    }
    value = sum_block(value);
    if (normalize) {
        __syncthreads(); // All lanes must consume the first reduction before scratch reuse.
        squared = sum_block(squared);
        value *= rsqrtf(squared / Hidden + 1e-6f);
    }
    if (!threadIdx.x)
        scores[(position * Candidates + predecessor) * Candidates + candidate] = value;
}

__device__ float adjusted(float value, int token, const SamplingConfig* configs,
                          int batch, const int* prefix, int length) {
    if (!configs) return value;
    const auto cfg = configs[batch];
    if (cfg.presence_penalty == 0 && cfg.frequency_penalty == 0) return value;
    int count = cfg.token_counts ? cfg.token_counts[token] : 0;
    for (int i = 0; i < length; ++i) count += prefix[i] == token;
    return value - (count > 0 ? cfg.presence_penalty : 0.0f) - cfg.frequency_penalty * count;
}

__global__ void path_kernel(const float* scores, const int* candidates, int width, int* drafts,
                            const SamplingConfig* configs) {
    const int batch = blockIdx.x;
    int predecessor = 0;
    for (int step = 0; step < width - 1; ++step) {
        const int position = batch * (width - 1) + step;
        const float* row   = scores + (position * Candidates + predecessor) * Candidates;
        int best           = 0;
        float best_value = adjusted(row[0], candidates[position * Candidates], configs, batch,
                                     drafts + batch * (width - 1), step);
        for (int c = 1; c < Candidates; ++c) {
            const float value = adjusted(row[c], candidates[position * Candidates + c], configs,
                                          batch, drafts + batch * (width - 1), step);
            if (value > best_value ||
                (value == best_value &&
                 candidates[position * Candidates + c] < candidates[position * Candidates + best]))
                { best = c; best_value = value; }
        }
        drafts[position] = candidates[position * Candidates + best];
        predecessor      = best;
    }
}

void matrix(const Tensor& value, DType dtype, int features, int rows, CudaExecutionView execution) {
    common::operand::dtype(value, dtype, "DSpine", "dtype mismatch");
    common::operand::shape(value, {features, rows, 1, 1}, "DSpine", "matrix");
    common::operand::contiguous_nonnull(value, "DSpine", "matrix", execution.device());
}

void width_rows(int width, int rows) {
    if (width < 2 || width > 16 || rows < width || rows > width * 8 || rows % width)
        throw std::invalid_argument("DSpine requires complete 2..16-token blocks and 1..8 rows");
}
} // namespace

void dspine_prepare_read(const Tensor& raw, Tensor& normalized, Tensor& rms, int width, float eps,
                         CudaExecutionView e) {
    const int rows = raw.ne[1];
    width_rows(width, rows);
    matrix(raw, DType::BF16, Hidden, rows, e);
    matrix(normalized, DType::BF16, Hidden, rows, e);
    matrix(rms, DType::FP32, rows, 1, e);
    if (!std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("DSpine read epsilon must be positive");
    read_kernel<<<rows, Threads, 0, e.stream()>>>(static_cast<const Bf16*>(raw.data),
                                                  static_cast<Bf16*>(normalized.data),
                                                  static_cast<float*>(rms.data), eps);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_normalize_codes(Tensor& codes, const Tensor& table, const Tensor& anchors, int width,
                            float eps, CudaExecutionView e) {
    const int rows = codes.ne[1];
    width_rows(width, rows);
    matrix(codes, DType::BF16, Code, rows, e);
    matrix(table, DType::BF16, Code, table.ne[1], e);
    matrix(anchors, DType::I32, rows / width, 1, e);
    normalize_kernel<<<rows, Threads, 0, e.stream()>>>(
        static_cast<Bf16*>(codes.data), static_cast<const Bf16*>(table.data),
        static_cast<const int*>(anchors.data), width, eps);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_message(const Tensor& packed, const Tensor& bias, int width, Tensor& message,
                    CudaExecutionView e) {
    const int rows = packed.ne[1];
    width_rows(width, rows);
    matrix(packed, DType::BF16, 3 * Rank, rows, e);
    matrix(bias, DType::BF16, Rank, 1, e);
    matrix(message, DType::BF16, Rank, rows, e);
    message_kernel<<<rows, Threads, 0, e.stream()>>>(static_cast<const Bf16*>(packed.data),
                                                     static_cast<const Bf16*>(bias.data), width,
                                                     static_cast<Bf16*>(message.data));
    CUDA_CHECK(cudaGetLastError());
}

void dspine_finish(const Tensor& delta, const Tensor& rms, int width, Tensor& residual,
                   CudaExecutionView e) {
    const int rows = residual.ne[1];
    width_rows(width, rows);
    matrix(delta, DType::BF16, Hidden, rows, e);
    matrix(residual, DType::BF16, Hidden, rows, e);
    matrix(rms, DType::FP32, rows, 1, e);
    finish_kernel<<<rows, Threads, 0, e.stream()>>>(static_cast<const Bf16*>(delta.data),
                                                    static_cast<const float*>(rms.data), width,
                                                    static_cast<Bf16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

void dspine_sender_codes(const Tensor& table, const Tensor& candidates, const Tensor& anchors,
                         int width, Tensor& codes, CudaExecutionView e) {
    width_rows(width, width);
    const int positions = candidates.ne[1];
    if (positions % (width - 1))
        throw std::invalid_argument("DSpine candidate rows must contain complete requests");
    width_rows(width, positions / (width - 1) * width);
    matrix(candidates, DType::I32, Candidates, positions, e);
    matrix(table, DType::BF16, Code, table.ne[1], e);
    matrix(anchors, DType::I32, positions / (width - 1), 1, e);
    matrix(codes, DType::BF16, Code, positions * Candidates, e);
    sender_kernel<<<positions * Candidates, Threads, 0, e.stream()>>>(
        static_cast<const Bf16*>(table.data), static_cast<const int*>(candidates.data),
        static_cast<const int*>(anchors.data), width, static_cast<Bf16*>(codes.data));
    CUDA_CHECK(cudaGetLastError());
}

void dspine_refine_message(const Tensor& receiver, const Tensor& senders, const Tensor& bias,
                           int width, Tensor& message, CudaExecutionView e) {
    const int rows = receiver.ne[1];
    width_rows(width, rows);
    const int count = rows / width * (width - 1) * Candidates;
    matrix(receiver, DType::BF16, 3 * Rank, rows, e);
    matrix(senders, DType::BF16, 3 * Rank, count, e);
    matrix(bias, DType::BF16, Rank, 1, e);
    matrix(message, DType::BF16, Rank, count, e);
    refine_kernel<<<count, Threads, 0, e.stream()>>>(
        static_cast<const Bf16*>(receiver.data), static_cast<const Bf16*>(senders.data),
        static_cast<const Bf16*>(bias.data), width, static_cast<Bf16*>(message.data));
    CUDA_CHECK(cudaGetLastError());
}

void dspine_greedy_path(const Tensor& raw, const Tensor& rms, const Tensor& delta,
                        const Tensor& norm, const Weight& head, const Tensor& candidates, int width,
                        Tensor& scores, Tensor& drafts, CudaExecutionView e,
                        const SamplingConfig* configs) {
    const int rows = raw.ne[1];
    width_rows(width, rows);
    const int batch = rows / width, positions = batch * (width - 1);
    matrix(raw, DType::BF16, Hidden, rows, e);
    matrix(rms, DType::FP32, rows, 1, e);
    matrix(delta, DType::BF16, Hidden, Candidates * positions, e);
    matrix(norm, DType::BF16, Hidden, 1, e);
    matrix(candidates, DType::I32, Candidates, positions, e);
    matrix(scores, DType::FP32, Candidates * Candidates, positions, e);
    matrix(drafts, DType::I32, width - 1, batch, e);
    if (head.device != e.device() || head.qtype != QType::Q8_G32_FP16 || head.k != Hidden ||
        head.group != 32 || head.layout != QuantLayout::RowSplit || !head.qdata || !head.scales)
        throw std::invalid_argument("DSpine re-scoring requires a Q8 group-32 target head");
    score_kernel<<<dim3(Candidates, Candidates, positions), Threads, 0, e.stream()>>>(
        static_cast<const Bf16*>(raw.data), static_cast<const float*>(rms.data),
        static_cast<const Bf16*>(delta.data), static_cast<const Bf16*>(norm.data),
        static_cast<const signed char*>(head.qdata), static_cast<const __half*>(head.scales),
        static_cast<const int*>(candidates.data), width, static_cast<float*>(scores.data),
        configs != nullptr);
    CUDA_CHECK(cudaGetLastError());
    path_kernel<<<batch, 1, 0, e.stream()>>>(static_cast<const float*>(scores.data),
                                             static_cast<const int*>(candidates.data), width,
                                             static_cast<int*>(drafts.data), configs);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops

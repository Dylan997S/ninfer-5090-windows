#include "ninfer/ops/dspine.h"
#include "ops/common/host/operands.h"

#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
using Bf16            = __nv_bfloat16;
constexpr int Hidden  = 5120;
constexpr int Threads = 256;

__global__ void post_write_norm_kernel(const Bf16* raw, const Bf16* norm, float eps, Bf16* output) {
    __shared__ float sums[Threads];
    const int row = blockIdx.x;
    float sum     = 0;
    for (int h = threadIdx.x; h < Hidden; h += Threads) {
        const float value = __bfloat162float(raw[row * Hidden + h]);
        sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / Hidden + eps);
    for (int h = threadIdx.x; h < Hidden; h += Threads) {
        const Bf16 normalized = __float2bfloat16(__bfloat162float(raw[row * Hidden + h]) * inverse);
        output[row * Hidden + h] =
            __float2bfloat16(__bfloat162float(normalized) * __bfloat162float(norm[h]));
    }
}
} // namespace

void dspine_post_write_norm(const Tensor& raw, const Tensor& norm, float eps, Tensor& output,
                            CudaExecutionView execution) {
    constexpr const char* op = "DSpine post-write norm";
    const int rows           = raw.ne[1];
    if (rows < 2 || rows > 512 || !std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument(
            "DSpine post-write norm requires 2..512 rows and positive epsilon");
    const auto device = execution.device();
    common::operand::tensor(raw, DType::BF16, {Hidden, rows}, 2, op, "raw", device);
    common::operand::tensor(norm, DType::BF16, {Hidden}, 2, op, "norm", device);
    common::operand::tensor(output, DType::BF16, {Hidden, rows}, 2, op, "output", device);
    common::operand::disjoint(common::operand::range(norm, op), common::operand::range(output, op),
                              op);
    if (raw.data != output.data)
        common::operand::disjoint(common::operand::range(raw, op),
                                  common::operand::range(output, op), op);
    post_write_norm_kernel<<<rows, Threads, 0, execution.stream()>>>(
        static_cast<const Bf16*>(raw.data), static_cast<const Bf16*>(norm.data), eps,
        static_cast<Bf16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops

#include "ops/linear/pair/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Instance = Nvfp4ScheduleInstance<Schedule, 5120>;

void project(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    const auto operands = nvfp4_a16_operands(x, w);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(y.data), w.n};
    const LinearIdentityEpilogue epilogue;
    if (x.ne[1] == 1)
        return launch_nvfp4_a16_gemv<Instance<Nvfp4A16GemvSchedule<
            8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>>(
                operands, output, epilogue, stream);
    if (x.ne[1] <= 8)
        return launch_nvfp4_a16_sliced_k_mma<Instance<Nvfp4SlicedInstance<8, 4, 2>>>(
            operands, output, epilogue, stream);
    if (x.ne[1] <= 16)
        return launch_nvfp4_a16_sliced_k_mma<Instance<Nvfp4SlicedInstance<16, 4, 2>>>(
            operands, output, epilogue, stream);
    if (x.ne[1] <= 32)
        return launch_nvfp4_a16_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 2>>>(
            operands, output, epilogue, stream);
    return launch_nvfp4_a16_mma<Instance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>>(
        operands, output, epilogue, stream);
}
} // namespace

void launch_nvfp4_pair(const Tensor& first_input, const Tensor& second_input,
                       const Weight& first, const Weight& second, Tensor& first_out,
                       Tensor& second_out, cudaStream_t stream) {
    project(first_input, first, first_out, stream);
    project(second_input, second, second_out, stream);
}
} // namespace ninfer::ops::detail

// DSpine attention output: weight-only NVFP4, with BF16 residual accumulation.
#include "ops/linear/add/nvfp4/launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Instance = Nvfp4ScheduleInstance<Schedule, 4096>;
}

Nvfp4LinearAddPlan select_nvfp4_add_n5120_k4096(LinearPolicy, std::int32_t tokens) {
    if (tokens == 1)
        return {launch_nvfp4_add_gemv<Instance<Nvfp4A16GemvSchedule<
            8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>>};
    if (tokens <= 8)
        return {launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<8, 4, 2>>>};
    if (tokens <= 16)
        return {launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<16, 4, 2>>>};
    if (tokens <= 32)
        return {launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 2>>>};
    if (tokens <= 64)
        return {launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<64, 2, 2>>>};
    if (tokens <= 128)
        return {launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>>};
    return {launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>};
}
} // namespace ninfer::ops::detail

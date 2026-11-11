#include "core/cpu/system.h"

#include <cpuid.h>

#include <cstring>

namespace ninfer {

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return {};
    state.leaf1_ecx = ecx;
    if ((ecx & ((1U << 26) | (1U << 27))) == ((1U << 26) | (1U << 27))) {
        unsigned low, high;
        asm volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        state.xcr0 = (static_cast<std::uint64_t>(high) << 32) | low;
    }
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        const auto max_subleaf = eax;
        state.leaf7_0_ebx      = ebx;
        state.leaf7_0_ecx      = ecx;
        if (max_subleaf >= 1) {
            __cpuid_count(7, 1, eax, ebx, ecx, edx);
            state.leaf7_1_eax = eax;
        }
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    unsigned eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::memcpy(result.vendor.data(), &ebx, 4);
    std::memcpy(result.vendor.data() + 4, &edx, 4);
    std::memcpy(result.vendor.data() + 8, &ecx, 4);
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return result;
    const auto family = (eax >> 8) & 15U;
    result.family     = family + (family == 15 ? (eax >> 20) & 255U : 0);
    result.model    = ((eax >> 4) & 15U) | ((family == 6 || family == 15) ? (eax >> 12) & 240U : 0);
    result.stepping = eax & 15U;
    result.virtualized = (ecx & (1U << 31)) != 0;
    return result;
}

} // namespace ninfer

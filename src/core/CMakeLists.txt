# The CPU execution layer is x86-64 only, and upstream implements it against hwloc on Linux. The
# port supplies a native Windows implementation of the same core/cpu/system.h interface in
# cpu/system_windows.cpp (Win32 processor groups and MSVC intrinsics), so the architecture ceiling
# stands while the platform one widens. CMAKE_SYSTEM_PROCESSOR is empty under some Windows
# generators, so an 8-byte pointer stands in for it when it is.
set(NINFER_CPU_X86_64 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(NINFER_CPU_X86_64 TRUE)
elseif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(NINFER_CPU_X86_64 TRUE)
endif()
if(NOT NINFER_CPU_X86_64)
  message(FATAL_ERROR "NInfer CPU execution currently requires x86-64")
endif()

if(WIN32)
  set(NINFER_CPU_SYSTEM_SOURCE cpu/system_windows.cpp)
  set(NINFER_CORE_LINKAGE STATIC)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(NINFER_CPU_SYSTEM_SOURCE cpu/system.cpp)
  set(NINFER_CORE_LINKAGE SHARED)
else()
  message(FATAL_ERROR "NInfer CPU execution requires Linux x86-64 or Windows x86-64")
endif()

add_library(ninfer_cpu ${NINFER_CORE_LINKAGE}
  ${NINFER_CPU_SYSTEM_SOURCE}
  cpu/task_pool.cpp
)
ninfer_internal_includes(ninfer_cpu)
if(WIN32)
  # hwloc backs upstream's Linux implementation only; the Windows path needs no external library.
  target_link_libraries(ninfer_cpu PRIVATE Threads::Threads)
else()
  target_link_libraries(ninfer_cpu PRIVATE PkgConfig::HWLOC Threads::Threads)
endif()

if(NINFER_TARGET STREQUAL "rtx5090")
  if(WIN32)
    # cpu/x86.cpp is the GCC <cpuid.h> implementation of current_cpu_features/current_cpu_identity.
    # The port supplies those same symbols from cpu/system_windows.cpp (MSVC intrinsics), so the
    # GCC translation unit is Linux-only here; the executor itself is platform-neutral.
    add_library(ninfer_cpu_executor STATIC cpu/executor.cpp)
  else()
    add_library(ninfer_cpu_executor ${NINFER_CORE_LINKAGE} cpu/executor.cpp cpu/x86.cpp)
  endif()
  ninfer_internal_includes(ninfer_cpu_executor)
  target_link_libraries(ninfer_cpu_executor PUBLIC ninfer_cpu)
endif()

add_library(ninfer_core ${NINFER_CORE_LINKAGE}
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  device_memory.cpp
  dtype.cpp
  nf4.cpp
  io/file.cpp
  io/read_queue.cpp
  gdn_replay_records.cpp
  kda_replay_records.cpp
  cyclic_kv_cache.cpp
  context_pool.cpp
  paged_kv_cache.cpp
  paged_kv_geometry.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_library(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()

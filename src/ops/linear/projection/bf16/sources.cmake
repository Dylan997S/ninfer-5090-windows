target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k512.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1536_k1024.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")

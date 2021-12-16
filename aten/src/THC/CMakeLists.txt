set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

install(FILES
          THCAtomics.cuh
          THCDeviceUtils.cuh
          THCGenerateByteType.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

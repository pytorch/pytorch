FILE(GLOB Eigen_Core_arch_CUDA_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_CUDA_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/CUDA COMPONENT Devel
)

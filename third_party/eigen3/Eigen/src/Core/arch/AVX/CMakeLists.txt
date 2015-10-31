FILE(GLOB Eigen_Core_arch_AVX_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_AVX_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/AVX COMPONENT Devel
)

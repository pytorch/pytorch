FILE(GLOB Eigen_LU_arch_SRCS "*.h")

INSTALL(FILES
  ${Eigen_LU_arch_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/LU/arch COMPONENT Devel
  )

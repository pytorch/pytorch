FILE(GLOB Eigen_Core_arch_AltiVec_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_AltiVec_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/AltiVec COMPONENT Devel
)

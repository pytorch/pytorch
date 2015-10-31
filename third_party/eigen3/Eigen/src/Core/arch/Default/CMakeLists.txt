FILE(GLOB Eigen_Core_arch_Default_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_Default_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/Default COMPONENT Devel
)

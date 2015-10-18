FILE(GLOB Eigen_CXX11_TensorSymmetry_util_SRCS "*.h")

INSTALL(FILES
  ${Eigen_CXX11_TensorSymmetry_util_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11/src/TensorSymmetry/util COMPONENT Devel
  )

FILE(GLOB Eigen_CXX11_Tensor_SRCS "*.h")

INSTALL(FILES
  ${Eigen_CXX11_Tensor_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11/src/Tensor COMPONENT Devel
  )

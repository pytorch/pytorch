FILE(GLOB Eigen_CXX11_TensorSymmetry_SRCS "*.h")

INSTALL(FILES
  ${Eigen_CXX11_TensorSymmetry_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11/src/TensorSymmetry COMPONENT Devel
  )

add_subdirectory(util)

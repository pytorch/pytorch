set(Eigen_CXX11_HEADERS Core Tensor TensorSymmetry)

install(FILES
  ${Eigen_CXX11_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11 COMPONENT Devel
  )

add_subdirectory(src)

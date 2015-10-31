FILE(GLOB Eigen_Core_Product_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_Product_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/products COMPONENT Devel
  )

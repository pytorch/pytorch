FILE(GLOB Eigen_SparseCholesky_SRCS "*.h")

INSTALL(FILES
  ${Eigen_SparseCholesky_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/SparseCholesky COMPONENT Devel
  )

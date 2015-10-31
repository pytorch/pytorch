FILE(GLOB Eigen_SparseExtra_SRCS "*.h")

INSTALL(FILES
  ${Eigen_SparseExtra_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/SparseExtra COMPONENT Devel
  )

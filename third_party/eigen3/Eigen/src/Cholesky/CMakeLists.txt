FILE(GLOB Eigen_Cholesky_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Cholesky_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Cholesky COMPONENT Devel
  )

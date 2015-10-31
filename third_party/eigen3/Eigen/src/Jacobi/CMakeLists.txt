FILE(GLOB Eigen_Jacobi_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Jacobi_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Jacobi COMPONENT Devel
  )

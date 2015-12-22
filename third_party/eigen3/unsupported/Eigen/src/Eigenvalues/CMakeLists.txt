FILE(GLOB Eigen_Eigenvalues_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Eigenvalues_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/Eigenvalues COMPONENT Devel
  )

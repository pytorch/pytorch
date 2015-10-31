FILE(GLOB Eigen_Polynomials_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Polynomials_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/Polynomials COMPONENT Devel
  )

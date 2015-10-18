FILE(GLOB Eigen_IterativeSolvers_SRCS "*.h")

INSTALL(FILES
  ${Eigen_IterativeSolvers_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/IterativeSolvers COMPONENT Devel
  )

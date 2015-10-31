FILE(GLOB Eigen_IterativeLinearSolvers_SRCS "*.h")

INSTALL(FILES
  ${Eigen_IterativeLinearSolvers_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/IterativeLinearSolvers COMPONENT Devel
  )

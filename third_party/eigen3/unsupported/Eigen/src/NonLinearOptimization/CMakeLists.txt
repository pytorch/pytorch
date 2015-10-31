FILE(GLOB Eigen_NonLinearOptimization_SRCS "*.h")

INSTALL(FILES
  ${Eigen_NonLinearOptimization_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/NonLinearOptimization COMPONENT Devel
  )

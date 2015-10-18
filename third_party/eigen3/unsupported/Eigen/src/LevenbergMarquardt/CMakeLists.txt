FILE(GLOB Eigen_LevenbergMarquardt_SRCS "*.h")

INSTALL(FILES
  ${Eigen_LevenbergMarquardt_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/LevenbergMarquardt COMPONENT Devel
  )

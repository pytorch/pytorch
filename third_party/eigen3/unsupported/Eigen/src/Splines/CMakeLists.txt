FILE(GLOB Eigen_Splines_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Splines_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/Splines COMPONENT Devel
  )

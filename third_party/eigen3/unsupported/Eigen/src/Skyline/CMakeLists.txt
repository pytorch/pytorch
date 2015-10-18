FILE(GLOB Eigen_Skyline_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Skyline_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/Skyline COMPONENT Devel
  )

FILE(GLOB Eigen_plugins_SRCS "*.h")

INSTALL(FILES
  ${Eigen_plugins_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/plugins COMPONENT Devel
  )

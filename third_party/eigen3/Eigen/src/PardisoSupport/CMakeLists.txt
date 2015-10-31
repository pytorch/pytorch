FILE(GLOB Eigen_PardisoSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_PardisoSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/PardisoSupport COMPONENT Devel
  )

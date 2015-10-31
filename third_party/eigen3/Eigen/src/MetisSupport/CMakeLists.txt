FILE(GLOB Eigen_MetisSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_MetisSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/MetisSupport COMPONENT Devel
  )

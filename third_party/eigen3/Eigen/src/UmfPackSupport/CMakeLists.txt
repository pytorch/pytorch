FILE(GLOB Eigen_UmfPackSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_UmfPackSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/UmfPackSupport COMPONENT Devel
  )

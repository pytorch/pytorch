FILE(GLOB Eigen_CholmodSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_CholmodSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/CholmodSupport COMPONENT Devel
  )

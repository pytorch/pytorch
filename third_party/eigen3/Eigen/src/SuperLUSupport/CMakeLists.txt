FILE(GLOB Eigen_SuperLUSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_SuperLUSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/SuperLUSupport COMPONENT Devel
  )

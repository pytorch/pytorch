FILE(GLOB Eigen_PastixSupport_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_PastixSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/PaStiXSupport COMPONENT Devel
  )

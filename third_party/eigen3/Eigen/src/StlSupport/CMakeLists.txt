FILE(GLOB Eigen_StlSupport_SRCS "*.h")

INSTALL(FILES
  ${Eigen_StlSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/StlSupport COMPONENT Devel
  )

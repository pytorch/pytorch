FILE(GLOB Eigen_Geometry_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Geometry_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Geometry COMPONENT Devel
  )

ADD_SUBDIRECTORY(arch)

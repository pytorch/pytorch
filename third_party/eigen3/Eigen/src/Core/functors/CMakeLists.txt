FILE(GLOB Eigen_Core_Functor_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_Functor_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/functors COMPONENT Devel
  )

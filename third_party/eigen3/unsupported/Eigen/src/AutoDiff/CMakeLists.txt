FILE(GLOB Eigen_AutoDiff_SRCS "*.h")

INSTALL(FILES
  ${Eigen_AutoDiff_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/AutoDiff COMPONENT Devel
  )

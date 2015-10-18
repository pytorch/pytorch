FILE(GLOB Eigen_SparseCore_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_SparseCore_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/SparseCore COMPONENT Devel
  )

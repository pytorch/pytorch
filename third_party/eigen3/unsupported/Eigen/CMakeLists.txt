set(Eigen_HEADERS AdolcForward BVH IterativeSolvers MatrixFunctions MoreVectorization AutoDiff AlignedVector3 Polynomials
                  FFT NonLinearOptimization SparseExtra IterativeSolvers
                  NumericalDiff Skyline MPRealSupport OpenGLSupport KroneckerProduct Splines LevenbergMarquardt
   )

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen COMPONENT Devel
  )

add_subdirectory(src)
add_subdirectory(CXX11)
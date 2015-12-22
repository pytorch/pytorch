set(Eigen_HEADERS 
  AdolcForward
  AlignedVector3
  ArpackSupport
  AutoDiff
  BVH
  FFT
  IterativeSolvers 
  KroneckerProduct
  LevenbergMarquardt
  MatrixFunctions 
  MoreVectorization
  MPRealSupport
  NonLinearOptimization
  NumericalDiff
  OpenGLSupport
  Polynomials
  Skyline 
  SparseExtra
  Splines
  )

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen COMPONENT Devel
  )

add_subdirectory(src)
add_subdirectory(CXX11)
if(__EIGEN_BLAS_INCLUDED)
  return()
endif()
set(__EIGEN_BLAS_INCLUDED TRUE)

if(NOT INTERN_BUILD_MOBILE OR NOT INTERN_USE_EIGEN_BLAS)
  return()
endif()

##############################################################################
# Eigen BLAS is built together with Libtorch mobile.
# By default, it builds code from third-party/eigen/blas submodule.
##############################################################################

set(CAFFE2_THIRD_PARTY_ROOT ${PROJECT_SOURCE_DIR}/third_party)
set(EIGEN_BLAS_SRC_DIR "${CAFFE2_THIRD_PARTY_ROOT}/eigen/blas" CACHE STRING "Eigen BLAS source directory")

set(EigenBlas_SRCS
  ${EIGEN_BLAS_SRC_DIR}/single.cpp
  ${EIGEN_BLAS_SRC_DIR}/double.cpp
  ${EIGEN_BLAS_SRC_DIR}/complex_single.cpp
  ${EIGEN_BLAS_SRC_DIR}/complex_double.cpp
  ${EIGEN_BLAS_SRC_DIR}/xerbla.cpp
  ${EIGEN_BLAS_SRC_DIR}/f2c/srotm.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/srotmg.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/drotm.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/drotmg.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/lsame.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/dspmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/ssbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/chbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/sspmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/zhbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/chpmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/dsbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/zhpmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/dtbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/stbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/ctbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/ztbmv.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/d_cnjg.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/r_cnjg.c
  ${EIGEN_BLAS_SRC_DIR}/f2c/complexdots.c
)

add_library(eigen_blas ${EigenBlas_SRCS})

# We build static versions of eigen blas but link into a shared library, so they need PIC.
set_property(TARGET eigen_blas PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS eigen_blas
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

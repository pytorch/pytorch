#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorFastGetSet.hpp"
#else

static inline scalar_t THTensor_(fastGetLegacy1dNoScalars)(THTensor *self, int64_t x0) {
  return self->unsafe_data<scalar_t>()[x0*THTensor_strideLegacyNoScalars(self, 0)];
}

static inline scalar_t THTensor_(fastGet1d)(THTensor *self, int64_t x0) {
  return self->unsafe_data<scalar_t>()[x0*self->stride(0)];
}

static inline scalar_t THTensor_(fastGet2d)(THTensor *self, int64_t x0, int64_t x1) {
  return self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)];
}

static inline scalar_t THTensor_(fastGet3d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2) {
  return self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)];
}

static inline scalar_t THTensor_(fastGet4d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3) {
  return self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)];
}

static inline scalar_t THTensor_(fastGet5d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4) {
  return self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)+(x4)*self->stride(4)];
}

static inline void THTensor_(fastSet1d)(THTensor *self, int64_t x0, scalar_t value) {
  self->unsafe_data<scalar_t>()[x0*self->stride(0)] = value;
}

static inline void THTensor_(fastSet2d)(THTensor *self, int64_t x0, int64_t x1, scalar_t value) {
  self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)] = value;
}

static inline void THTensor_(fastSet3d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, scalar_t value) {
  self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)] = value;
}

static inline void THTensor_(fastSet4d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value) {
  self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)] = value;
}

static inline void THTensor_(fastSet5d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4, scalar_t value) {
  self->unsafe_data<scalar_t>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)+(x4)*self->stride(4)] = value;
}

#endif

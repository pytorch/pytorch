#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorFastGetSet.hpp"
#else

static inline real THTensor_(fastGet1d)(THTensor *self, int64_t x0) {
  return (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]];
}

static inline real THTensor_(fastGet2d)(THTensor *self, int64_t x0, int64_t x1) {
  return (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]];
}

static inline real THTensor_(fastGet3d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2) {
  return (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]];
}

static inline real THTensor_(fastGet4d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3) {
  return (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]+(x3)*self->stride[3]];
}

static inline real THTensor_(fastGet5d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4) {
  return (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]+(x3)*self->stride[3]+(x4)*self->stride[4]];
}

static inline void THTensor_(fastSet1d)(THTensor *self, int64_t x0, real value) {
  (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]] = value;
}

static inline void THTensor_(fastSet2d)(THTensor *self, int64_t x0, int64_t x1, real value) {
  (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]] = value;
}

static inline void THTensor_(fastSet3d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, real value) {
  (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]] = value;
}

static inline void THTensor_(fastSet4d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value) {
  (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]+(x3)*self->stride[3]] = value;
}

static inline void THTensor_(fastSet5d)(THTensor *self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4, real value) {
  (THStorage_(data)(self->storage)+self->storageOffset)[(x0)*self->stride[0]+(x1)*self->stride[1]+(x2)*self->stride[2]+(x3)*self->stride[3]+(x4)*self->stride[4]] = value;
}

#endif

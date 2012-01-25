#ifndef TH_TENSOR_MACROS_INC
#define TH_TENSOR_MACROS_INC

/* fast method to access to tensor data */

#define THTensor_fastGet1d(self, x0)                                    \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]])

#define THTensor_fastGet2d(self, x0, x1)                                \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]])

#define THTensor_fastGet3d(self, x0, x1, x2)                            \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]])

#define THTensor_fastGet4d(self, x0, x1, x2, x3)                        \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]+(x3)*(self)->stride[3]])

#define THTensor_fastSet1d(self, x0, value)                             \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]] = value)

#define THTensor_fastSet2d(self, x0, x1, value)                         \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]] = value)

#define THTensor_fastSet3d(self, x0, x1, x2, value)                     \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]] = value)

#define THTensor_fastSet4d(self, x0, x1, x2, x3, value)                 \
  (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]+(x3)*(self)->stride[3]] = value)

#endif

#include "c10/macros/Macros.h"

#ifdef CAFFE2_BUILD_OBSERVER_LIB
#define CAFFE2_OBSERVER_API C10_EXPORT
#else
#define CAFFE2_OBSERVER_API C10_IMPORT
#endif

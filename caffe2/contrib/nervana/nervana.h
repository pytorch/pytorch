#ifndef CAFFE2_FB_NERVANA_INIT_H_
#define CAFFE2_FB_NERVANA_INIT_H_

#include "caffe2/core/init.h"
#include "caffe2/core/flags.h"

#include "nervana_c_api.h"

/**
 * A flag that specifies the nervana cubin path.
 */
CAFFE2_DECLARE_string(nervana_cubin_path);

namespace caffe2 {

/**
 * An empty class to be used in identifying the engine in the math functions.
 */
class NervanaEngine {};

/**
 * Returns whether the nervana kernels are loaded or not.
 */
bool NervanaKernelLoaded();

/**
 * An initialization function that is run once by caffe2::GlobalInit()
 * that initializes the nervana kernels.
 */
bool Caffe2InitializeNervanaKernels();

}  // namespace caffe2

#endif  // CAFFE2_FB_NERVANA_INIT_H_

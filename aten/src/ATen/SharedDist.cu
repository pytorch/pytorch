#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/Config.h"

#include <nvfunctional>
 
namespace at {
  namespace native {
    namespace dist {
      template<typename precision_t>
      struct baseSampler {
        nvstd::function<precision_t(void)> sampler;
        baseSampler(nvstd::function<precision_t(void)> sampler): sampler(sampler) {}
        precision_t sample() {
          return sampler();
        }
      };
    }
  }
}

// this version is only linked if CUDA is enabled, so we can safely just use CUDA features here

#include "snpe_ffi.h"
#include <atomic>
#include <mutex>

namespace caffe2 {

static std::once_flag flag;
std::string& gSNPELocation() {
  static std::string g_snpe_location;
  std::call_once(flag, [](){
    g_snpe_location = "";
  });
  return g_snpe_location;
}

}


#include "caffe2/core/typeid.h"

#include <map>

namespace caffe2 {
namespace internal {

std::map<TypeId, string> g_caffe2_type_name_map;

}  // namespace internal
}  // namespace caffe2

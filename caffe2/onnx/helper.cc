#include "helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 { namespace onnx  {

size_t DummyName::counter_ = 0;

std::unordered_set<std::string>& DummyName::get_used_names() {
  static std::unordered_set<std::string> used_names;
  return used_names;
}

std::string DummyName::NewDummyName() {
  while (true) {
    const std::string name = caffe2::MakeString("OC2_DUMMY_", counter_++);
    auto ret = get_used_names().insert(name);
    if (ret.second) {
      return name;
    }
  }
}

void DummyName::Reset(const std::unordered_set<std::string> &used_names) {
  auto& names = get_used_names();
  names = used_names;
  counter_ = 0;
}

}}

#include "caffe2/python/pybind_state_detail.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/logging.h"
#include "caffe2/python/pybind_state_fetcher_feeder.h"

namespace caffe2 {
namespace python {

namespace python_detail {

// Python Op implementations.
using FuncRegistry = std::unordered_map<std::string, Func>;

FuncRegistry& gRegistry() {
  // Always leak the objects registered here.
  static FuncRegistry* r = new FuncRegistry();
  return *r;
}

const Func& getOpFunc(const std::string& token) {
  CAFFE_ENFORCE(
      gRegistry().count(token),
      "Python operator for ",
      token,
      " is not available. If you use distributed training it probably means "
      "that python implementation has to be registered in each of the workers");
  return gRegistry()[token];
}

const Func& getGradientFunc(const std::string& token) {
  return getOpFunc(token + "_gradient");
}

py::object fetchBlob(Workspace* ws, const std::string& name) {
  CAFFE_ENFORCE(ws->HasBlob(name), "Can't find blob: ", name);
  const caffe2::Blob& blob = *(ws->GetBlob(name));
  auto fetcher = CreateFetcher(blob.meta().id());
  if (fetcher) {
    return fetcher->Fetch(blob);
  } else {
    // If there is no fetcher registered, return a metainfo string.
    // If all branches failed, we will return a metainfo string.
    std::stringstream ss;
    ss << caffe2::string(name) << ", a C++ native class of type "
       << blob.TypeName() << ".";
    return py::bytes(ss.str());
  }
}
} // namespace python_detail

} // namespace python
} // namespace caffe2

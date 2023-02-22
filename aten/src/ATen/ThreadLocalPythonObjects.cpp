#include <c10/core/TensorImpl.h>
#include <ATen/ThreadLocalPythonObjects.h>
#include <c10/util/Exception.h>

namespace at {
namespace impl {

static thread_local ThreadLocalPythonObjects py_objects;


void ThreadLocalPythonObjects::set(std::string key, std::shared_ptr<SafePyObject> value) {
  py_objects.obj_dict_[key] = value;
}

const std::shared_ptr<SafePyObject>& ThreadLocalPythonObjects::get(std::string key) {
  TORCH_CHECK(py_objects.obj_dict_.count(key));
  return py_objects.obj_dict_[key];
}

bool ThreadLocalPythonObjects::contains(std::string key) {
  return py_objects.obj_dict_.count(key);
}

void ThreadLocalPythonObjects::set_state(const ThreadLocalPythonObjects& state) {
  py_objects = state;
}

const ThreadLocalPythonObjects& ThreadLocalPythonObjects::get_state() {
  return py_objects;
}


}
}

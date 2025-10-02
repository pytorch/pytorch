#include <c10/core/TensorImpl.h>
#include <ATen/ThreadLocalPythonObjects.h>
#include <c10/util/Exception.h>

#include <utility>

namespace at::impl {

static thread_local ThreadLocalPythonObjects py_objects;


void ThreadLocalPythonObjects::set(const std::string& key, std::shared_ptr<SafePyObject> value) {
  py_objects.obj_dict_[key] = std::move(value);
}

const std::shared_ptr<SafePyObject>& ThreadLocalPythonObjects::get(const std::string& key) {
  TORCH_CHECK(py_objects.obj_dict_.count(key));
  return py_objects.obj_dict_[key];
}

bool ThreadLocalPythonObjects::contains(const std::string& key) {
  return py_objects.obj_dict_.count(key);
}

void ThreadLocalPythonObjects::set_state(ThreadLocalPythonObjects state) {
  py_objects = std::move(state);
}

const ThreadLocalPythonObjects& ThreadLocalPythonObjects::get_state() {
  return py_objects;
}


} // namespace at::impl

#include "caffe2/core/logging.h"
#include "caffe2/core/module.h"

#ifndef _MSC_VER
#include <dlfcn.h>
#endif

namespace caffe2 {

static std::mutex& gModuleChangeMutex() {
  static std::mutex m_;
  return m_;
}

static CaffeMap<string, const ModuleSchema*>& MutableCurrentModules() {
  static CaffeMap<string, const ModuleSchema*> module_schema_map_;
  return module_schema_map_;
}

// Note(jiayq): I am not sure whether the module handles are going to be used
// as C2 uses modules via registration, but let's keep the handles at least.
static CaffeMap<string, void*> CurrentModuleHandles() {
  static CaffeMap<string, void*> module_handle_map_;
  return module_handle_map_;
}

const CaffeMap<string, const ModuleSchema*>& CurrentModules() {
  return MutableCurrentModules();
}

ModuleSchema::ModuleSchema(const char* name, const char* description)
    : name_(name), description_(description) {
  std::lock_guard<std::mutex> guard(gModuleChangeMutex());
  MutableCurrentModules().emplace(name, this);
}

bool HasModule(const string& name) {
 auto& modules = CurrentModules();
 return (modules.find(name) != modules.end());
}

#ifdef _MSC_VER

void LoadModule(const string& name, const string& filename) {
  CAFFE_ENFORCE(!HasModule(name),
    "On Windows, LoadModule is currently not supported yet and you should "
    "use static linking for any module that you intend to use.");
}

#else

void LoadModule(const string& name, const string& filename) {
  CAFFE_ENFORCE(
      name.size() > 0 || filename.size() > 0,
      "You must provide at least one of name and filename.");
  if (name.size() && HasModule(name)) {
    VLOG(1) << "Module " << name << " already present. Skip loading."; 
    return;
  }
  void* handle = nullptr;
  if (filename.size()) {
    handle = dlopen(
        filename.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CAFFE_ENFORCE(handle != nullptr,
      "Cannot load module ",
      name,
      " (with given filename ",
      filename,
      "), are you sure it is correct?");
  } else {
    string inferred_name = string("lib") + name + ".so";
    handle = dlopen(
        inferred_name.c_str(), RTLD_NOW | RTLD_GLOBAL);
#ifdef __APPLE__
    // For apple, we will also try the dylib extension.
    if (!handle) {
      string inferred_name = string("lib") + name + ".dylib";
      handle = dlopen(
          inferred_name.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    }
#endif
    CAFFE_ENFORCE(handle != nullptr,
        "Cannot load module ",
        name,
        " (with inferred filename ",
        inferred_name,
        "), are you sure it is in the dynamic linker search path?");
  }
  // After the module is loaded, we should check if it actually has the
  // intended module name. If not, it might be that the module file name
  // and the module name are inconsistent.
  if (name.size()) {
    string module_name_check = "gCaffe2ModuleSanityCheck" + name;
    CAFFE_ENFORCE(dlsym(handle, module_name_check.c_str()),
        "The loaded module ",
        name,
        " did not pass the module name sanity check. Is it built with the "
        "right configs? Make sure the file name and the CAFFE2_MODULE name "
        "are consistent.");
    // After it passes the dlopen and dlsym check, we should add it to the
    // current handles.
    std::lock_guard<std::mutex> guard(gModuleChangeMutex());
    CurrentModuleHandles()[name] = handle;
  } else {
    // If not, we issue a warning that one is recommended to use explicit
    // module name.
    LOG(WARNING)
        << "Module file " << filename
        << " was loaded without a proper module name. It is recommended "
           "that one load a model with an explicit module name in addition "
           "to the filename.";
    // As a contingency, we will store the current module handle with the
    // filename.
    std::lock_guard<std::mutex> guard(gModuleChangeMutex());
    CurrentModuleHandles()[filename] = handle;
  }
}

#endif // _MSC_VER

}  // namespace caffe2


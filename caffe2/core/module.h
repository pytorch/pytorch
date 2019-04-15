/**
 * A global dictionary that holds information about what Caffe2 modules have
 * been loaded in the current runtime, and also utility functions to load
 * modules.
 */
#ifndef CAFFE2_CORE_MODULE_H_
#define CAFFE2_CORE_MODULE_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>

#include "caffe2/core/common.h"
#include <c10/util/typeid.h>

namespace caffe2 {

/**
 * A module schema that can be used to store specific information about
 * different modules. Currently, we only store the name and a simple
 * description of what this module does.
 */
class CAFFE2_API ModuleSchema {
 public:
  ModuleSchema(const char* name, const char* description);

 private:
  const char* name_;
  const char* description_;
};


/**
 * @brief Current Modules present in the Caffe2 runtime.
 * Returns:
 *   map: a map of modules and (optionally) their description. The key is the
 *       module name, and the value is the description for that module. The
 *       module name is recommended to be the part that constitutes the trunk
 *       of the dynamic library: for example, a module called
 *       libcaffe2_db_rocksdb.so should have the name "caffe2_db_rocksdb". The
 *       reason we do not use "lib" is because it's somewhat redundant, and
 *       the reason we do not include ".so" is for cross-platform compatibility
 *       on platforms like mac os.
 */
CAFFE2_API const CaffeMap<string, const ModuleSchema*>& CurrentModules();

/**
 * @brief Checks whether a module is already present in the current binary.
 */
CAFFE2_API bool HasModule(const string& name);

/**
 * @brief Load a module.
 * Inputs:
 *   name: a module name or a path name.
 *       It is recommended that you use the name of the module, and leave the
 *       full path option to only experimental modules.
 *   filename: (optional) a filename that serves as a hint to load the module.
 */
CAFFE2_API void LoadModule(const string& name, const string& filename="");


#define CAFFE2_MODULE(name, description)                                    \
  extern "C" {                                                              \
    bool gCaffe2ModuleSanityCheck##name() { return true; }                  \
  }                                                                         \
  namespace {                                                               \
    static ::caffe2::ModuleSchema module_schema_##name(#name, description); \
  }

}  // namespace caffe2
#endif  // CAFFE2_CORE_MODULE_H_

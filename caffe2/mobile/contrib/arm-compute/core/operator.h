#ifndef CAFFE2_OPENGL_OPERATOR_H_
#define CAFFE2_OPENGL_OPERATOR_H_

#include "caffe2/core/operator.h"
#include "caffe2/core/registry.h"

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(GLOperatorRegistry, OperatorBase, const OperatorDef &,
                       Workspace *);
#define REGISTER_GL_OPERATOR_CREATOR(key, ...)                                 \
  CAFFE_REGISTER_CREATOR(GLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_GL_OPERATOR(name, ...)                                        \
  extern void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                  \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_GL##name() {              \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                            \
  }                                                                            \
  CAFFE_REGISTER_CLASS(GLOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_GL_OPERATOR_STR(str_name, ...)                                \
  CAFFE_REGISTER_TYPED_CLASS(GLOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_GL_OPERATOR_WITH_ENGINE(name, engine, ...)                    \
  CAFFE_REGISTER_CLASS(GLOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

} // namespace caffe2

#endif // CAFFE2_OPENGL_OPERATOR_H_

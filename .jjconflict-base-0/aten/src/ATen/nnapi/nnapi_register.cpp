#include <ATen/nnapi/nnapi_bind.h>

// Set flag if running on ios
#ifdef __APPLE__
  #include <TargetConditionals.h>
  #if TARGET_OS_IPHONE
    #define IS_IOS_NNAPI_BIND
  #endif
#endif

#ifndef IS_IOS_NNAPI_BIND
TORCH_LIBRARY(_nnapi, m) {
  m.class_<torch::nnapi::bind::NnapiCompilation>("Compilation")
    .def(torch::jit::init<>())
    .def("init", &torch::nnapi::bind::NnapiCompilation::init)
    .def("init2", &torch::nnapi::bind::NnapiCompilation::init2)
    .def("run", &torch::nnapi::bind::NnapiCompilation::run)
    ;
}
#else
  #undef IS_IOS_NNAPI_BIND
#endif

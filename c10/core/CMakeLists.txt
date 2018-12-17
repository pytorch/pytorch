set(SOURCES
  impl/DeviceGuardImplInterface.cpp
  Allocator.cpp
  CopyBytes.cpp
  DefaultDtype.cpp
  Device.cpp
  DeviceType.cpp
  Scalar.cpp
  Storage.cpp
  StorageImpl.cpp
  Stream.cpp
  TensorImpl.cpp
  TensorOptions.cpp
  TensorTypeId.cpp
  TensorTypeIdRegistration.cpp
  UndefinedTensorImpl.cpp
)

add_library(c10_core STATIC
  ${SOURCES}
)

c10_setup_visibility(c10_core)

target_link_libraries(c10_core PUBLIC c10_util)

install(TARGETS c10_core EXPORT Caffe2Targets DESTINATION lib)

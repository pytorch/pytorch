set(HEADERS
  dispatch/DeviceId.h
  dispatch/Dispatcher.h
  dispatch/DispatchKey.h
  dispatch/DispatchTable.h
  dispatch/KernelRegistration.h
  dispatch/LayoutId.h
  dispatch/OpSchema.h
  dispatch/OpSchemaRegistration.h
  impl/DeviceGuardImplInterface.h
  impl/FakeGuardImpl.h
  impl/InlineDeviceGuard.h
  impl/InlineStreamGuard.h
  impl/VirtualGuardImpl.h
  opschema/layer_norm.h
  Allocator.h
  Backend.h
  CopyBytes.h
  DefaultDtype.h
  DefaultTensorOptions.h
  Device.h
  DeviceGuard.h
  DeviceType.h
  Layout.h
  Scalar.h
  ScalarType.h
  ScalarTypeUtils.h
  Storage.h
  StorageImpl.h
  Stream.h
  StreamGuard.h
  Tensor.h
  TensorImpl.h
  TensorOptions.h
  TensorTypeId.h
  TensorTypeIdRegistration.h
  UndefinedTensorImpl.h
  WrapDimMinimal.h
)

set(SOURCES
  dispatch/DeviceId.cpp
  dispatch/Dispatcher.cpp
  dispatch/DispatchKey.cpp
  dispatch/DispatchTable.cpp
  dispatch/KernelRegistration.cpp
  dispatch/LayoutId.cpp
  dispatch/OpSchema.cpp
  dispatch/OpSchemaRegistration.cpp
  impl/DeviceGuardImplInterface.cpp
  opschema/layer_norm.cpp
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
  ${HEADERS}
  ${SOURCES}
)

c10_setup_visibility(c10_core)

target_link_libraries(c10_core PUBLIC c10_util)

install(TARGETS c10_core EXPORT Caffe2Targets DESTINATION lib)

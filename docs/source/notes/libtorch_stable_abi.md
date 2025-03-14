# LibTorch Stable ABI

This note will eventually contain more details on how to use the APIs in torch/csrc/stable. For the moment, it contains a table of internal representations:
1. type in custom extension: type used within the end user custom library.
2. StableIValue representation: a stable conversion of the type to liaison between the user model vs libtorch.so.
3. type in libtorch: type used within libtorch.so.
4. Schema Type: type as described by the schema, which we hail as the source of truth for both ATen ops in native_functions.yaml and for user defined custom ops registered to the dispatcher.

|  type in custom extension    |   StableIValue representation   |   type in libtorch  |   Schema Type  |
| -------- | ------- | ------- | ------- |
| std::optional\<S> | \*reinterpret_cast\<(StableIValue\*)\*>, pointer to a StableIValue recursively defined | std::optional\<T> | OptionalType |
| std::nullopt | \*reinterpret_cast\<nullptr_t\*> | IValue() | NoneType |
| RAIIATH | \*reinterpret_cast\<uint64_t\*> of AtenTensorHandle | at::Tensor |  TensorType |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::ScalarType | ScalarTypeType |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::Layout | LayoutType |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::MemoryFormat | MemoryFormatType |
| bool | \*reinterpret_cast\<uint64_t\*> | bool | BoolType |
| int64_t | \*reinterpret_cast\<uint64_t\*> | int64_t | IntType |
| float | \*reinterpret_cast\<uint64_t\*> | double | FloatType |
| ? | ? | c10::Device | DeviceObjType |
| ? | ? | c10::Stream | StreamObjType |
| ? | ? | c10::complex<double> | ComplexType |
| ? | ? | at::Scalar | NumberType |
| ? | ? | std::string/const char*/ivalue::ConstantString | StringType |
| ? | ? | at::Storage | StorageType |
| ? | ? | at::Generator | GeneratorType |
| ? | ? | c10::List\<T> | AnyListType / ListType |
| ? | ? | ivalue::Tuple\<T> | AnyTupleType / TupleType |
| ? | ? | at::Quantizer | QuantizerType |
| ? | ? | ivalue::EnumHolder | AnyEnumType / EnumType |
| ? | ? | c10::SymInt | SymIntType |
| ? | ? | c10::SymFloat | SymFloatType |
| ? | ? | c10::SymBool | SymBoolType |
| ? | ? | at::QScheme | QSchemeType |
| ? | ? | c10::Dict<IValue, IValue> | DictType |
| ? | ? | c10::RRefInterface | RRefType |
| ? | ? | ivalue::Object | ClassType / AnyClassType |
| ? | ? | torch::jit::Module | Module |
| ? | ? | torch::jit::Function* | FunctionType |
| ? | ? | c10::Capsule | CapsuleType |
| ? | ? | ivalue::PyObjectHolder | PyObjectType |
| ? | ? | ivalue::Future | FutureType |
| ? | ? | ivalue::Await | AwaitType |
| ? | ? | ? | AnyType |
| ? | ? | ? | VarType |
| ? | ? | ? | InterfaceType |
| ? | ? | ? | UnionType |
| ? | ? | ? | DynamicType |

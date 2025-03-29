# LibTorch Stable ABI

This note will eventually contain more details on how to use the APIs in torch/csrc/stable. For the moment, it contains a table of internal representations:
1. type in custom extension: type used within the end user custom library.
2. StableIValue representation: a stable conversion of the type to liaison between the user model vs libtorch.so in an ABI-stable manner.
3. type in libtorch: type used within libtorch.so (or any code binary locked with libtorch).
4. Schema Type: type as described by the schema, which we hail as the source of truth for both ATen ops in native_functions.yaml and for user defined custom operators registered to the dispatcher via TORCH_LIBRARY or torch.library.

|  type in custom extension    |   StableIValue representation   |   type in libtorch  |   Schema Type  |
| -------- | ------- | ------- | ------- |
| std::optional\<S> | \*reinterpret_cast\<(StableIValue\*)\*>, pointer to a StableIValue recursively defined | std::optional\<T> | Type? |
| std::nullopt | \*reinterpret_cast\<nullptr_t\*> | IValue() | None |
| RAIIATH | \*reinterpret_cast\<uint64_t\*> of AtenTensorHandle | at::Tensor |  Tensor |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::ScalarType | ScalarType |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::Layout | Layout |
| int32_t | \*reinterpret_cast\<uint64_t\*> | at::MemoryFormat | MemoryFormat |
| bool | \*reinterpret_cast\<uint64_t\*> | bool | bool |
| int64_t | \*reinterpret_cast\<uint64_t\*> | int64_t | int |
| double | \*reinterpret_cast\<uint64_t\*> | double | float |
| ? | ? | c10::Device | Device |
| ? | ? | c10::Stream | Stream |
| ? | ? | c10::complex<double> | complex |
| ? | ? | at::Scalar | Scalar |
| ? | ? | std::string/const char*/ivalue::ConstantString | str |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| ? | ? | c10::List\<T> | Type[] |
| ? | ? | ivalue::Tuple\<T> | (Type, ...) |
| ? | ? | c10::SymInt | SymInt |
| ? | ? | c10::SymFloat | SymFloat |
| ? | ? | c10::SymBool | SymBool |
| ? | ? | at::QScheme | QScheme |

Our confidently supported types are the ones in the table that have completed rows. For a limited set of use cases, we also implicitly support any literal type that is representable within 64 bits as StableIValues, as the default reinterpret_cast will succeed. You can work with StableIValue abstractions in your custom kernel for types such as c10::Device even if there is no standard defined representation of device in custom extensions. For example, a custom operator can take as argument a StableIValue device and directly pass it through to an aten operator with aoti_torch_call_dispatcher.

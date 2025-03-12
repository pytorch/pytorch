## LibTorch Stable ABI

todo: write a real intro

todo: write a paragraph about ownership

This README.md should eventually become a real doc on how to use the APIs in torch/csrc/stable. For the moment it will contain a big table of extension type -> StableIValue representation -> libtorch type -> IValue

|  type in custom extension    |   StableIValue representation   |   type in libtorch  |   IValue  | 
| -------- | ------- | ------- | ------- |
| std::optional\<S> | ? | std::optional\<T> | Optional|
| RAIIATH | AtenTensorHandle | at::Tensor |  Tensor |
| ? | ? | c10::Device | Device |
| void* | ? | c10::Stream | Stream |
| int32_t | *reinterpret_cast\<*uint64_t> | at::ScalarType | ScalarType | 
| int32_t | *reinterpret_cast\<*uint64_t> | at::Layout | Layout |
| int32_t | *reinterpret_cast\<*uint64_t> | at::MemoryFormat | MemoryFormat |
| ? | ? | at::Scalar | Scalar |
| float | *reinterpret_cast\<*uint64_t> | double | Double |
| ? | ? | c10::complex<double> | ComplexDouble |
| int64_t | *reinterpret_cast\<*uint64_t> | int64_t | Int |
| bool | *reinterpret_cast\<*uint64_t> | bool | Bool |
| ? | ? | std::string/const char*/ivalue::ConstantString | String |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| ? | ? | c10::List<T> | List |
| ? | ? | std::vector<T> | Vector |
| ? | ? | at::DimVector | DimVector |
| ? | ? | ivalue::Tuple | Tuple |
| ? | ? | at::Quantizer | Quantizer |
| ? | ? | ivalue::EnumHolder | Enum |
| ? | ? | c10::SymInt | SymInt |
| ? | ? | c10::SymFloat | SymFloat |
| ? | ? | c10::SymBool | SymBool |
| ? | ? | at::DimName | DimName
| ? | ? | caffe2::Blob | Blob |
| ? | ? | at::QScheme | QScheme |


|  type in custom extension    |   StableIValue representation   |  less libtorchy types  | IValue |
| -------- | ------- | ------- | ------- |
| | | c10::Dict<IValue, IValue> | GenericDict |
| | | c10::RRefInterface | RRef |
| | | ivalue::Object | Object |
| | | torch::jit::Module | Module | 
| | | torch::CustomClassHolder | Capsule/CustomClass |
| | | ivalue::PyObjectHolder | PyObject |
| | | IValue() | Uninitialized |
| | | ivalue::Future | Future |
| | | ivalue::Await | Await 



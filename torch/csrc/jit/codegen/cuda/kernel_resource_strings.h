namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// IO data structure for kernel code;
static auto code_template_tensor_struct = R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;

template<typename T, int N>
struct Tensor {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};
)";

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
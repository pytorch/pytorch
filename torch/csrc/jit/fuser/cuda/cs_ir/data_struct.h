STRINGIFY(
template<typename T>
struct IO_struct {
  long int shapes[8];
  long int strides[8];
  T* data;
};
)

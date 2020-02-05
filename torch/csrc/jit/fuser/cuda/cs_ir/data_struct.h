STRINGIFY(
template<typename T, typename INDEX_T, int DIM>
struct IO_struct {
  INDEX_T shapes[DIM];
  INDEX_T strides[DIM];
  T* data;
};
)

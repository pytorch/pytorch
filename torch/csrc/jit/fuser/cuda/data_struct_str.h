STRINGIFY(
template<typename T, typename INDEX_T, int DIM>
struct CudaTensor {
public:
  INDEX_T size(int i) {
    return sizes_[i];
  };

  INDEX_T stride(int i) {
    return strides_[i];
  };

  T& operator()(INDEX_T ind) {
    return data[ind];
  };

protected:
  INDEX_T sizes_[DIM];
  INDEX_T strides_[DIM];
  T* data;
};
)

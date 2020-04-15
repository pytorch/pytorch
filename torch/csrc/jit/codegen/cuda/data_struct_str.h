STRINGIFY(template <typename T> struct Tensor {
 public:
  T& operator[](int ind) {
    return data[ind];
  };

  int size[8];
  int stride[8];
  T* data;
  int nDim;
};)

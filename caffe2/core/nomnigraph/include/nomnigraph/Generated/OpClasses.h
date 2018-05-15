class Relu : public NeuralNetOperator {
 public:
  Relu() : NeuralNetOperator(NNKind::Relu) {}

  ~Relu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Relu);

 private:
};

class Conv : public NeuralNetOperator {
 public:
  Conv(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1},
      int group = 1,
      vector<int> dilations = {1, 1})
      : NeuralNetOperator(NNKind::Conv),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides),
        Group(group),
        Dilations(dilations) {}

  ~Conv() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Conv);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  int getGroup() const {
    return Group;
  }

  vector<int> getDilations() const {
    return Dilations;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

  void setGroup(int group) {
    Group = group;
  }

  void setDilations(vector<int> dilations) {
    Dilations = dilations;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
  int Group;
  vector<int> Dilations;
};

class ConvRelu : public NeuralNetOperator {
 public:
  ConvRelu(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1},
      int group = 1,
      vector<int> dilations = {1, 1})
      : NeuralNetOperator(NNKind::ConvRelu),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides),
        Group(group),
        Dilations(dilations) {}

  ConvRelu(const Conv& conv)
      : NeuralNetOperator(NNKind::ConvRelu),
        KernelShape(conv.getKernelShape()),
        Pads(conv.getPads()),
        Strides(conv.getStrides()),
        Group(conv.getGroup()),
        Dilations(conv.getDilations()) {}

  ~ConvRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ConvRelu);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  int getGroup() const {
    return Group;
  }

  vector<int> getDilations() const {
    return Dilations;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

  void setGroup(int group) {
    Group = group;
  }

  void setDilations(vector<int> dilations) {
    Dilations = dilations;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
  int Group;
  vector<int> Dilations;
};

class ConvTranspose : public NeuralNetOperator {
 public:
  ConvTranspose(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1},
      int group = 1,
      vector<int> dilations = {1, 1})
      : NeuralNetOperator(NNKind::ConvTranspose),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides),
        Group(group),
        Dilations(dilations) {}

  ~ConvTranspose() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ConvTranspose);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  int getGroup() const {
    return Group;
  }

  vector<int> getDilations() const {
    return Dilations;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

  void setGroup(int group) {
    Group = group;
  }

  void setDilations(vector<int> dilations) {
    Dilations = dilations;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
  int Group;
  vector<int> Dilations;
};

class AveragePool : public NeuralNetOperator {
 public:
  AveragePool(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::AveragePool),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides) {}

  ~AveragePool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(AveragePool);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
};

class AveragePoolRelu : public NeuralNetOperator {
 public:
  AveragePoolRelu(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::AveragePoolRelu),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides) {}

  AveragePoolRelu(const AveragePool& averagePool)
      : NeuralNetOperator(NNKind::AveragePoolRelu),
        KernelShape(averagePool.getKernelShape()),
        Pads(averagePool.getPads()),
        Strides(averagePool.getStrides()) {}

  ~AveragePoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(AveragePoolRelu);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
};

class MaxPool : public NeuralNetOperator {
 public:
  MaxPool(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::MaxPool),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides) {}

  ~MaxPool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(MaxPool);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
};

class MaxPoolRelu : public NeuralNetOperator {
 public:
  MaxPoolRelu(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::MaxPoolRelu),
        KernelShape(kernelShape),
        Pads(pads),
        Strides(strides) {}

  MaxPoolRelu(const MaxPool& maxPool)
      : NeuralNetOperator(NNKind::MaxPoolRelu),
        KernelShape(maxPool.getKernelShape()),
        Pads(maxPool.getPads()),
        Strides(maxPool.getStrides()) {}

  ~MaxPoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(MaxPoolRelu);

  vector<int> getKernelShape() const {
    return KernelShape;
  }

  vector<int> getPads() const {
    return Pads;
  }

  vector<int> getStrides() const {
    return Strides;
  }

  void setKernelShape(vector<int> kernelShape) {
    KernelShape = kernelShape;
  }

  void setPads(vector<int> pads) {
    Pads = pads;
  }

  void setStrides(vector<int> strides) {
    Strides = strides;
  }

 private:
  vector<int> KernelShape;
  vector<int> Pads;
  vector<int> Strides;
};

class Sum : public NeuralNetOperator {
 public:
  Sum() : NeuralNetOperator(NNKind::Sum) {}

  ~Sum() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Sum);

 private:
};

class SumRelu : public NeuralNetOperator {
 public:
  SumRelu() : NeuralNetOperator(NNKind::SumRelu) {}

  SumRelu(const Sum& sum) : NeuralNetOperator(NNKind::SumRelu) {}

  ~SumRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(SumRelu);

 private:
};

class Send : public NeuralNetOperator {
 public:
  Send(string destination)
      : NeuralNetOperator(NNKind::Send), Destination(destination) {}

  ~Send() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Send);

  string getDestination() const {
    return Destination;
  }

  void setDestination(string destination) {
    Destination = destination;
  }

 private:
  string Destination;
};

class Receive : public NeuralNetOperator {
 public:
  Receive(string source) : NeuralNetOperator(NNKind::Receive), Source(source) {}

  ~Receive() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Receive);

  string getSource() const {
    return Source;
  }

  void setSource(string source) {
    Source = source;
  }

 private:
  string Source;
};

class BatchNormalization : public NeuralNetOperator {
 public:
  BatchNormalization(
      float epsilon = 1e-5f,
      float momentum = 0.9f,
      bool spatial = true,
      bool isTest = false)
      : NeuralNetOperator(NNKind::BatchNormalization),
        Epsilon(epsilon),
        Momentum(momentum),
        Spatial(spatial),
        IsTest(isTest) {}

  ~BatchNormalization() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(BatchNormalization);

  float getEpsilon() const {
    return Epsilon;
  }

  float getMomentum() const {
    return Momentum;
  }

  bool getSpatial() const {
    return Spatial;
  }

  bool getIsTest() const {
    return IsTest;
  }

  void setEpsilon(float epsilon) {
    Epsilon = epsilon;
  }

  void setMomentum(float momentum) {
    Momentum = momentum;
  }

  void setSpatial(bool spatial) {
    Spatial = spatial;
  }

  void setIsTest(bool isTest) {
    IsTest = isTest;
  }

 private:
  float Epsilon;
  float Momentum;
  bool Spatial;
  bool IsTest;
};

class FC : public NeuralNetOperator {
 public:
  FC() : NeuralNetOperator(NNKind::FC) {}

  ~FC() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(FC);

 private:
};

class GivenTensorFill : public NeuralNetOperator {
 public:
  GivenTensorFill() : NeuralNetOperator(NNKind::GivenTensorFill) {}

  ~GivenTensorFill() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(GivenTensorFill);

 private:
};

class Concat : public NeuralNetOperator {
 public:
  Concat() : NeuralNetOperator(NNKind::Concat) {}

  ~Concat() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Concat);

 private:
};

class Softmax : public NeuralNetOperator {
 public:
  Softmax() : NeuralNetOperator(NNKind::Softmax) {}

  ~Softmax() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Softmax);

 private:
};

class ChannelShuffle : public NeuralNetOperator {
 public:
  ChannelShuffle() : NeuralNetOperator(NNKind::ChannelShuffle) {}

  ~ChannelShuffle() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ChannelShuffle);

 private:
};

class Add : public NeuralNetOperator {
 public:
  Add() : NeuralNetOperator(NNKind::Add) {}

  ~Add() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Add);

 private:
};

class Reshape : public NeuralNetOperator {
 public:
  Reshape() : NeuralNetOperator(NNKind::Reshape) {}

  ~Reshape() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Reshape);

 private:
};

class Flatten : public NeuralNetOperator {
 public:
  Flatten() : NeuralNetOperator(NNKind::Flatten) {}

  ~Flatten() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Flatten);

 private:
};

class Int8Quantize : public NeuralNetOperator {
 public:
  Int8Quantize() : NeuralNetOperator(NNKind::Int8Quantize) {}

  ~Int8Quantize() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Quantize);

 private:
};

class Int8Dequantize : public NeuralNetOperator {
 public:
  Int8Dequantize() : NeuralNetOperator(NNKind::Int8Dequantize) {}

  ~Int8Dequantize() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Dequantize);

 private:
};

class Int8AveragePool : public NeuralNetOperator {
 public:
  Int8AveragePool() : NeuralNetOperator(NNKind::Int8AveragePool) {}

  Int8AveragePool(const AveragePool& averagePool)
      : NeuralNetOperator(NNKind::Int8AveragePool) {}

  ~Int8AveragePool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8AveragePool);

 private:
};

class Int8Conv : public NeuralNetOperator {
 public:
  Int8Conv() : NeuralNetOperator(NNKind::Int8Conv) {}

  Int8Conv(const Conv& conv) : NeuralNetOperator(NNKind::Int8Conv) {}

  ~Int8Conv() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Conv);

 private:
};

class Int8ConvTranspose : public NeuralNetOperator {
 public:
  Int8ConvTranspose() : NeuralNetOperator(NNKind::Int8ConvTranspose) {}

  Int8ConvTranspose(const ConvTranspose& convTranspose)
      : NeuralNetOperator(NNKind::Int8ConvTranspose) {}

  ~Int8ConvTranspose() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8ConvTranspose);

 private:
};

class Int8FC : public NeuralNetOperator {
 public:
  Int8FC() : NeuralNetOperator(NNKind::Int8FC) {}

  Int8FC(const FC& fC) : NeuralNetOperator(NNKind::Int8FC) {}

  ~Int8FC() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8FC);

 private:
};

class Int8MaxPool : public NeuralNetOperator {
 public:
  Int8MaxPool() : NeuralNetOperator(NNKind::Int8MaxPool) {}

  Int8MaxPool(const MaxPool& maxPool)
      : NeuralNetOperator(NNKind::Int8MaxPool) {}

  ~Int8MaxPool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8MaxPool);

 private:
};

class Int8Relu : public NeuralNetOperator {
 public:
  Int8Relu() : NeuralNetOperator(NNKind::Int8Relu) {}

  Int8Relu(const Relu& relu) : NeuralNetOperator(NNKind::Int8Relu) {}

  ~Int8Relu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Relu);

 private:
};

class Int8GivenTensorFill : public NeuralNetOperator {
 public:
  Int8GivenTensorFill() : NeuralNetOperator(NNKind::Int8GivenTensorFill) {}

  Int8GivenTensorFill(const GivenTensorFill& givenTensorFill)
      : NeuralNetOperator(NNKind::Int8GivenTensorFill) {}

  ~Int8GivenTensorFill() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8GivenTensorFill);

 private:
};

class Int8Concat : public NeuralNetOperator {
 public:
  Int8Concat() : NeuralNetOperator(NNKind::Int8Concat) {}

  Int8Concat(const Concat& concat) : NeuralNetOperator(NNKind::Int8Concat) {}

  ~Int8Concat() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Concat);

 private:
};

class Int8Softmax : public NeuralNetOperator {
 public:
  Int8Softmax() : NeuralNetOperator(NNKind::Int8Softmax) {}

  Int8Softmax(const Softmax& softmax)
      : NeuralNetOperator(NNKind::Int8Softmax) {}

  ~Int8Softmax() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Softmax);

 private:
};

class Int8ChannelShuffle : public NeuralNetOperator {
 public:
  Int8ChannelShuffle() : NeuralNetOperator(NNKind::Int8ChannelShuffle) {}

  Int8ChannelShuffle(const ChannelShuffle& channelShuffle)
      : NeuralNetOperator(NNKind::Int8ChannelShuffle) {}

  ~Int8ChannelShuffle() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8ChannelShuffle);

 private:
};

class Int8Sum : public NeuralNetOperator {
 public:
  Int8Sum() : NeuralNetOperator(NNKind::Int8Sum) {}

  Int8Sum(const Sum& sum) : NeuralNetOperator(NNKind::Int8Sum) {}

  ~Int8Sum() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Sum);

 private:
};

class Int8Add : public NeuralNetOperator {
 public:
  Int8Add() : NeuralNetOperator(NNKind::Int8Add) {}

  Int8Add(const Add& add) : NeuralNetOperator(NNKind::Int8Add) {}

  ~Int8Add() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Add);

 private:
};

class Int8Reshape : public NeuralNetOperator {
 public:
  Int8Reshape() : NeuralNetOperator(NNKind::Int8Reshape) {}

  Int8Reshape(const Reshape& reshape)
      : NeuralNetOperator(NNKind::Int8Reshape) {}

  ~Int8Reshape() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Reshape);

 private:
};

class Int8Flatten : public NeuralNetOperator {
 public:
  Int8Flatten() : NeuralNetOperator(NNKind::Int8Flatten) {}

  Int8Flatten(const Flatten& flatten)
      : NeuralNetOperator(NNKind::Int8Flatten) {}

  ~Int8Flatten() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8Flatten);

 private:
};

class Int8ConvRelu : public NeuralNetOperator {
 public:
  Int8ConvRelu() : NeuralNetOperator(NNKind::Int8ConvRelu) {}

  Int8ConvRelu(const ConvRelu& convRelu)
      : NeuralNetOperator(NNKind::Int8ConvRelu) {}

  ~Int8ConvRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8ConvRelu);

 private:
};

class Int8SumRelu : public NeuralNetOperator {
 public:
  Int8SumRelu() : NeuralNetOperator(NNKind::Int8SumRelu) {}

  Int8SumRelu(const SumRelu& sumRelu)
      : NeuralNetOperator(NNKind::Int8SumRelu) {}

  ~Int8SumRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8SumRelu);

 private:
};

class Int8AveragePoolRelu : public NeuralNetOperator {
 public:
  Int8AveragePoolRelu() : NeuralNetOperator(NNKind::Int8AveragePoolRelu) {}

  Int8AveragePoolRelu(const AveragePoolRelu& averagePoolRelu)
      : NeuralNetOperator(NNKind::Int8AveragePoolRelu) {}

  ~Int8AveragePoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8AveragePoolRelu);

 private:
};

class Int8MaxPoolRelu : public NeuralNetOperator {
 public:
  Int8MaxPoolRelu() : NeuralNetOperator(NNKind::Int8MaxPoolRelu) {}

  Int8MaxPoolRelu(const MaxPoolRelu& maxPoolRelu)
      : NeuralNetOperator(NNKind::Int8MaxPoolRelu) {}

  ~Int8MaxPoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Int8MaxPoolRelu);

 private:
};

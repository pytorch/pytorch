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
  Concat(int axis = -1, bool addAxis = false)
      : NeuralNetOperator(NNKind::Concat), Axis(axis), AddAxis(addAxis) {}

  ~Concat() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Concat);

  int getAxis() const {
    return Axis;
  }

  bool getAddAxis() const {
    return AddAxis;
  }

  void setAxis(int axis) {
    Axis = axis;
  }

  void setAddAxis(bool addAxis) {
    AddAxis = addAxis;
  }

 private:
  int Axis;
  bool AddAxis;
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

class NCHW2NHWC : public NeuralNetOperator {
 public:
  NCHW2NHWC() : NeuralNetOperator(NNKind::NCHW2NHWC) {}

  ~NCHW2NHWC() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(NCHW2NHWC);

 private:
};

class NHWC2NCHW : public NeuralNetOperator {
 public:
  NHWC2NCHW() : NeuralNetOperator(NNKind::NHWC2NCHW) {}

  ~NHWC2NCHW() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(NHWC2NCHW);

 private:
};

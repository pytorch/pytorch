
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
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides),
        group_(group),
        dilations_(dilations) {}

  ~Conv() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Conv);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  int getGroup() const {
    return group_;
  }

  vector<int> getDilations() const {
    return dilations_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

  void setGroup(int group) {
    group_ = group;
  }

  void setDilations(vector<int> dilations) {
    dilations_ = dilations;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
  int group_;
  vector<int> dilations_;
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
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides),
        group_(group),
        dilations_(dilations) {}

  ConvRelu(const Conv& conv)
      : NeuralNetOperator(NNKind::ConvRelu),
        kernelShape_(conv.getKernelShape()),
        pads_(conv.getPads()),
        strides_(conv.getStrides()),
        group_(conv.getGroup()),
        dilations_(conv.getDilations()) {}

  ~ConvRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ConvRelu);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  int getGroup() const {
    return group_;
  }

  vector<int> getDilations() const {
    return dilations_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

  void setGroup(int group) {
    group_ = group;
  }

  void setDilations(vector<int> dilations) {
    dilations_ = dilations;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
  int group_;
  vector<int> dilations_;
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
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides),
        group_(group),
        dilations_(dilations) {}

  ~ConvTranspose() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ConvTranspose);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  int getGroup() const {
    return group_;
  }

  vector<int> getDilations() const {
    return dilations_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

  void setGroup(int group) {
    group_ = group;
  }

  void setDilations(vector<int> dilations) {
    dilations_ = dilations;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
  int group_;
  vector<int> dilations_;
};

class AveragePool : public NeuralNetOperator {
 public:
  AveragePool(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::AveragePool),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides) {}

  ~AveragePool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(AveragePool);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
};

class AveragePoolRelu : public NeuralNetOperator {
 public:
  AveragePoolRelu(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::AveragePoolRelu),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides) {}

  AveragePoolRelu(const AveragePool& averagePool)
      : NeuralNetOperator(NNKind::AveragePoolRelu),
        kernelShape_(averagePool.getKernelShape()),
        pads_(averagePool.getPads()),
        strides_(averagePool.getStrides()) {}

  ~AveragePoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(AveragePoolRelu);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
};

class MaxPool : public NeuralNetOperator {
 public:
  MaxPool(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::MaxPool),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides) {}

  ~MaxPool() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(MaxPool);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
};

class MaxPoolRelu : public NeuralNetOperator {
 public:
  MaxPoolRelu(
      vector<int> kernelShape,
      vector<int> pads = {0, 0},
      vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::MaxPoolRelu),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides) {}

  MaxPoolRelu(const MaxPool& maxPool)
      : NeuralNetOperator(NNKind::MaxPoolRelu),
        kernelShape_(maxPool.getKernelShape()),
        pads_(maxPool.getPads()),
        strides_(maxPool.getStrides()) {}

  ~MaxPoolRelu() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(MaxPoolRelu);

  vector<int> getKernelShape() const {
    return kernelShape_;
  }

  vector<int> getPads() const {
    return pads_;
  }

  vector<int> getStrides() const {
    return strides_;
  }

  void setKernelShape(vector<int> kernelShape) {
    kernelShape_ = kernelShape;
  }

  void setPads(vector<int> pads) {
    pads_ = pads;
  }

  void setStrides(vector<int> strides) {
    strides_ = strides;
  }

 private:
  vector<int> kernelShape_;
  vector<int> pads_;
  vector<int> strides_;
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
      : NeuralNetOperator(NNKind::Send), destination_(destination) {}

  ~Send() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Send);

  string getDestination() const {
    return destination_;
  }

  void setDestination(string destination) {
    destination_ = destination;
  }

 private:
  string destination_;
};

class Receive : public NeuralNetOperator {
 public:
  Receive(string source)
      : NeuralNetOperator(NNKind::Receive), source_(source) {}

  ~Receive() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Receive);

  string getSource() const {
    return source_;
  }

  void setSource(string source) {
    source_ = source;
  }

 private:
  string source_;
};

class BatchNormalization : public NeuralNetOperator {
 public:
  BatchNormalization(
      float epsilon = 1e-5f,
      float momentum = 0.9f,
      bool spatial = true,
      bool isTest = false)
      : NeuralNetOperator(NNKind::BatchNormalization),
        epsilon_(epsilon),
        momentum_(momentum),
        spatial_(spatial),
        isTest_(isTest) {}

  ~BatchNormalization() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(BatchNormalization);

  float getEpsilon() const {
    return epsilon_;
  }

  float getMomentum() const {
    return momentum_;
  }

  bool getSpatial() const {
    return spatial_;
  }

  bool getIsTest() const {
    return isTest_;
  }

  void setEpsilon(float epsilon) {
    epsilon_ = epsilon;
  }

  void setMomentum(float momentum) {
    momentum_ = momentum;
  }

  void setSpatial(bool spatial) {
    spatial_ = spatial;
  }

  void setIsTest(bool isTest) {
    isTest_ = isTest;
  }

 private:
  float epsilon_;
  float momentum_;
  bool spatial_;
  bool isTest_;
};

class Clip : public NeuralNetOperator {
 public:
  Clip(float min, float max)
      : NeuralNetOperator(NNKind::Clip), min_(min), max_(max) {}

  ~Clip() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Clip);

  float getMin() const {
    return min_;
  }

  float getMax() const {
    return max_;
  }

  void setMin(float min) {
    min_ = min;
  }

  void setMax(float max) {
    max_ = max;
  }

 private:
  float min_;
  float max_;
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
      : NeuralNetOperator(NNKind::Concat), axis_(axis), addAxis_(addAxis) {}

  ~Concat() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Concat);

  int getAxis() const {
    return axis_;
  }

  bool getAddAxis() const {
    return addAxis_;
  }

  void setAxis(int axis) {
    axis_ = axis;
  }

  void setAddAxis(bool addAxis) {
    addAxis_ = addAxis;
  }

 private:
  int axis_;
  bool addAxis_;
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
  Add(int broadcast = 0)
      : NeuralNetOperator(NNKind::Add), broadcast_(broadcast) {}

  ~Add() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Add);

  int getBroadcast() const {
    return broadcast_;
  }

  void setBroadcast(int broadcast) {
    broadcast_ = broadcast;
  }

 private:
  int broadcast_;
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

class CopyToOpenCL : public NeuralNetOperator {
 public:
  CopyToOpenCL() : NeuralNetOperator(NNKind::CopyToOpenCL) {}

  ~CopyToOpenCL() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(CopyToOpenCL);

 private:
};

class CopyFromOpenCL : public NeuralNetOperator {
 public:
  CopyFromOpenCL() : NeuralNetOperator(NNKind::CopyFromOpenCL) {}

  ~CopyFromOpenCL() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(CopyFromOpenCL);

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

class Declare : public NeuralNetOperator {
 public:
  Declare() : NeuralNetOperator(NNKind::Declare) {}

  ~Declare() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Declare);

 private:
};

class Export : public NeuralNetOperator {
 public:
  Export() : NeuralNetOperator(NNKind::Export) {}

  ~Export() {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Export);

 private:
};

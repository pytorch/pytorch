
#pragma once

#include "GLImage.h"
#include "caffe2/core/net.h"
#include "caffe2/core/predictor.h"

namespace caffe2 {
class GLPredictor : public PredictorBase {
 public:
  using TensorVector = PredictorBase::TensorVector;
  using TensorMap = PredictorBase::TensorMap;
  using OutputTensorVector = PredictorBase::OutputTensorVector;

  GLPredictor(const NetDef& init_net,
              const NetDef& run_net,
              bool use_texture_input = false,
              Workspace* parent = nullptr);

  template <class T>
  bool run(std::vector<GLImageVector<T>*>& inputs, std::vector<const GLImageVector<T>*>* outputs);

  virtual bool run(const TensorVector& inputs, OutputTensorVector& outputs, bool threadsafe = false) { return false; }
  virtual bool run_map(const TensorMap& inputs, OutputTensorVector& outputs, bool threadsafe = false) { return false; }


  ~GLPredictor();
};
} // namespace caffe2

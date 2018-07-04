
#pragma once

#include "GLImage.h"
#include "caffe2/core/net.h"
#include "caffe2/core/predictor.h"

namespace caffe2 {
class GLPredictor : public Predictor {
 public:
  GLPredictor(const NetDef& init_net,
              const NetDef& run_net,
              bool use_texture_input = false,
              Workspace* parent = nullptr);

  template <class T>
  bool run(std::vector<GLImageVector<T>*>& inputs, std::vector<const GLImageVector<T>*>* outputs);

  ~GLPredictor();
};
} // namespace caffe2

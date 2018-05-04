#pragma once

#include <torch/nn/module.h>

#include <cstdint>

namespace torch { namespace nn {
class Conv : public torch::nn::CloneableModule<Conv> {
 private:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan)
      : Nd_(Nd),
        in_channels_(in_chan),
        out_channels_(out_chan),
        stride_(makeTup(1, 1)),
        padding_(makeTup(0)),
        dilation_(makeTup(1, 1)),
        dilated_(false),
        output_padding_(makeTup(0)) {}

 public:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, int ks)
      : Conv(Nd, in_chan, out_chan) {
    ks_ = makeTup(ks, 1);
  }

  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, IntVec ks)
      : Conv(Nd, in_chan, out_chan) {
    ks_ = makeTup(ks);
  }

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  template <typename T>
  Conv& stride(T s) {
    stride_ = makeTup(s, 1);
    return *this;
  }
  template <typename T>
  Conv& padding(T s) {
    padding_ = makeTup(s);
    return *this;
  }
  template <typename T>
  Conv& dilation(T s) {
    dilation_ = makeTup(s, 1);
    return *this;
  }
  template <typename T>
  Conv& output_padding(T s) {
    output_padding_ = makeTup(s);
    return *this;
  }

  AUTOGRAD_KWARG(Conv, bool, transposed, false, true)
  AUTOGRAD_KWARG(Conv, bool, no_bias, false, true)
  AUTOGRAD_KWARG(Conv, int, groups, 1, 1)

  Variable weight, bias;
  uint32_t Nd_;
  uint32_t in_channels_;
  uint32_t out_channels_;
  IntVec ks_;
  IntVec stride_;
  IntVec padding_;
  IntVec dilation_;
  bool dilated_;
  IntVec output_padding_;

 protected:
  IntVec makeTup(int x, int def = 0) {
    IntVec ret;
    if (Nd_ == 1) {
      ret.push_back(x);
      ret.push_back(def);
    } else {
      for (auto i = 0U; i < Nd_; i++)
        ret.push_back(x);
    }
    return ret;
  }
  IntVec makeTup(IntVec x) {
    return x;
  }
};

class Conv1d : public Conv {
 public:
  Conv1d(uint32_t i, uint32_t o, int ks) : Conv(1, i, o, ks) {}
  Conv1d(uint32_t i, uint32_t o, IntVec ks) : Conv(1, i, o, ks) {}
};

class Conv2d : public Conv {
 public:
  Conv2d(uint32_t i, uint32_t o, int ks) : Conv(2, i, o, ks) {}
  Conv2d(uint32_t i, uint32_t o, IntVec ks) : Conv(2, i, o, ks) {}
};

class Conv3d : public Conv {
 public:
  Conv3d(uint32_t i, uint32_t o, int ks) : Conv(3, i, o, ks) {}
  Conv3d(uint32_t i, uint32_t o, IntVec ks) : Conv(3, i, o, ks) {}
};
}} // namespace torch::nn

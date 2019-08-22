#include <caffe2/ideep/operators/conv_pool_base_op.h>
#include <iostream>
namespace caffe2 {

class IDEEPInt8ConvOp : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        //scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {
    //OPERATOR_NEEDS_FEATURE(pad_l() == pad_r() && pad_t() == pad_b(),
    //                       "Uneven padding not supported.");
    fusion_type_ = FUSION_UNKNOWN;
    last_input_ = BIAS_OR_INPUT_S;
    algo_ = ialgo::convolution_direct;
    auto conv_algorithm = OperatorBase::GetSingleArgument<int>(
        "conv_algorithm", CONV_ALGORITHM_AUTO);
    if (conv_algorithm == CONV_ALGORITHM_WINOGRAD) {
      algo_ = ialgo::convolution_winograd;
    }
    CAFFE_ENFORCE(zero_point_ == 128 || zero_point_ == 0);
    
    int per_channel_output = this->template GetSingleArgument<int>("per_channel_output", 0);
    if (per_channel_output) {
      scale_ = OperatorBase::GetRepeatedArgument<float>("Y_scale", {1.0});
    } else {
      scale_ = {(this->template GetSingleArgument<float>("Y_scale", 1.0))};
    }
   
    Y_scales_ = ConvertScales(scale_);
    expect_x_format = ideep::format::nhwc;
#ifdef USE_EULER      
    need_euler_ = OperatorBase::GetSingleArgument<int>("need_euler", 0);
    if (need_euler_) {
      std::string euler_order_src = OperatorBase::GetSingleArgument<string>("euler_order_src", "nChw16c");
      if (euler_order_src == "nChw16c") {
        expect_x_format = ideep::format::nChw16c;
      }
      std::string euler_order_dst = OperatorBase::GetSingleArgument<string>("euler_order_dst", "nChw16c");
      if (euler_order_dst == "nChw16c") {
	eparam.dst_format = ideep::format::nChw16c;
      } else {
	eparam.dst_format = ideep::format::nhwc;
      }
      eparam.shared_workspace_key = operator_def.name();
      eparam.nthreads = OperatorBase::GetSingleArgument<int>("nthreads", 0);
      eparam.tile_size = OperatorBase::GetSingleArgument<int>("tile_size", 5);
      eparam.conv_algorithm = OperatorBase::GetSingleArgument<int>("conv_algorithm", 0);
      eparam.execution_mode = execution_mode_trans(
              OperatorBase::GetSingleArgument<string>("execution_mode", "0x0000"));
      eparam.dst_type = OperatorBase::GetSingleArgument<string>("dst_type", "u8");
      eparam.scratch_pad = NULL;
      eparam.flatting = OperatorBase::GetRepeatedArgument<int>("flatting", {1, 1});
      eparam.blocking = OperatorBase::GetRepeatedArgument<int>("blocking", {1, 1});
      eparam.partition = OperatorBase::GetRepeatedArgument<int>("partition", {1, 1});
      eparam.wino_tinput_quant = OperatorBase::GetRepeatedArgument<float>("wino_tinput_quant", {1.0, 0.0});
      int X_zero_point = OperatorBase::GetSingleArgument<int>("X_zero_point", 0);
      eparam.input_quant = {1.0, float(X_zero_point)};
      int sum_zero_point = OperatorBase::GetSingleArgument<int>("sum_zero_point", 0);
      eparam.sum_quant = {1.0, float(sum_zero_point)};
      eparam.with_argmax = bool(OperatorBase::GetSingleArgument<int>("with_argmax", 0));
      X_scales_ = 0.0;
    }
#endif
  }
  virtual ~IDEEPInt8ConvOp() {}

  int execution_mode_trans(string execution_mode_) {
    int execution_mode_tr = 0x0000;
    if (execution_mode_ == "0xa160") {
      execution_mode_tr = 0xa160;
    } else if (execution_mode_ == "0xc160") {
      execution_mode_tr = 0xc160;
    } else if (execution_mode_ == "0xd160") {
      execution_mode_tr = 0xd160;
    } else if (execution_mode_ == "0xa161") {
      execution_mode_tr = 0xa161;
    } else if (execution_mode_ == "0xa133") {
      execution_mode_tr = 0xa133;
    } else if (execution_mode_ == "0xb161") {
      execution_mode_tr = 0xb161;
    } else {
      printf("No support execution_mode");
    }
    return execution_mode_tr;
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const auto &X = Input(INPUT_X);
    const auto &filter = Input(FILTER);
    auto *Y = Output(OUTPUT);
    //if (!Y_dims[0]) {
    //  Y_dims = CalcOutputDims(X, filter.get_dim(0));
    //}
    CAFFE_ENFORCE(X.has_scale());
    CAFFE_ENFORCE(4 == X.ndims() && 4 == filter.ndims());
    CAFFE_ENFORCE(X.get_data_type() == idtype::s8
        || X.get_data_type() == idtype::u8);
    CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1), " is not equal to kernel channels * group:",
        filter.get_dim(1), "*", group_);

    if (X.get_internal_format() != expect_x_format) {
      X_in.init({X.get_dims(), X.get_data_type(), expect_x_format});
      X_in.set_scale(X.get_scale());
      X_in.feed_from_src(X);
    } else {
      X_in = X;
    }

    bool input_changed = (cached_X_descriptor_ != X_in.get_descriptor());
    if (input_changed) {
      op_key_.clear();
      cached_X_descriptor_ = X_in.dup_descriptor();
      Y_dims = CalcOutputDims(X, filter.get_dim(0));
    }
#ifdef USE_EULER      
    if (need_euler_) {
      CAFFE_ENFORCE(X.get_data_type() == idtype::u8);
      if (!X_scales_) { 
	X_scales_ = 1/X.get_scale()[0];
        eparam.input_quant[0] = X_scales_;
      }
      //float max_scale = scale_[0];
      //for (int i  = 0; i < scale_.size(); ++i) {
      //  if (max_scale < scale_[i])
      //    max_scale = scale_[i];
      //}
      eparam.output_quant = {scale_[0], float(zero_point_)};
      
      bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor());
      if (weights_changed) {
        op_key_.clear();
        cached_weights_descriptor_ = filter.dup_descriptor();

        auto filter_in = filter.as_weights();
        filter_in.make_group(group_);
	ideep::format filter_format = ideep::format::OIhw16i16o;
        ideep::tensor::descriptor expected_descriptor = {filter_in.get_dims(), idtype::f32, filter_format};
        if (filter.get_descriptor() != expected_descriptor) {
          filter_.init(expected_descriptor);
          filter_.feed_from(filter_in);
        } else {
          filter_ = filter_in;
        }

        if (InputSize() > last_input_) {
          // NOTE: If the bias is shared by other operators in this module,
          // The existing bias scale should not satisfy current operator.
          // Thus, we have to requantize it by current input and filter scales.
          auto bias = Input(BIAS_OR_INPUT_S);
          bias_.init({bias.get_dims(), idtype::f32, ideep::format::x});
          if (bias.get_descriptor() != bias_.get_descriptor()) {
            bias_.feed_from(bias);
          } else {
            bias_ = bias;
          }
	}
      }

      if (InputSize() > last_input_) {
        ideep::convolution_forward::compute(
            op_key_, comp, X_in, filter_, bias_, Y_dims, *Y,
            stride_, dilation_, pad_tl(), pad_br(), group_,
            eparam, X_in.get_scale(), iscale(), Y_scales_, attr_,
            algo_, iprop::forward_inference, ipadding::zero);
      } else {
        ideep::convolution_forward::compute(
            op_key_, comp, X_in, filter_, Y_dims, *Y,
            stride_, dilation_, pad_tl(), pad_br(), group_,
            eparam, X_in.get_scale(), iscale(), Y_scales_, attr_,
            algo_, iprop::forward_inference, ipadding::zero);
      }

      if (X.get_dim(2)==1 && X.get_dim(3) ==1) {
	if (eparam.with_argmax ) {
	  Y->set_descriptor({{X.get_dim(0),1,1,1}, idtype::s32, ideep::format::nhwc});
	} else {
	  Y->set_descriptor({Y_dims, Y->get_data_type(), ideep::format::nhwc});
	}
      }
      if (fusion_type_ != FUSION_CONV_RELU && 
              fusion_type_ != FUSION_CONV_BRELU &&
              fusion_type_ != FUSION_UNKNOWN) {
        CAFFE_ENFORCE(
            Y == &(Input(InputSize() - 1)),
            "Convolution fusion op: InPlace is enforced for sum fusion.");
      }

      return true;
    }
#endif
    bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor()) || 
      (algo_ == ialgo::convolution_winograd && input_changed);
    if (weights_changed) {
      op_key_.clear();
      cached_weights_descriptor_ = filter.dup_descriptor();
      //CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

      itensor filter_in;
      auto X_dt = X.get_data_type();
      lowp_kind_ = ilowp_kind::LOWP_U8S8;
      auto filter_scale = filter.get_scale();
      if (X_dt == idtype::s8) {
        lowp_kind_ = ilowp_kind::LOWP_S8S8;
        filter_in = filter.as_weights().to_public();
      } else {
        filter_in = filter.as_weights();
      }
      filter_in.make_group(group_);

      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter_in.get_dims(), idtype::s8, stride_, pad_tl(), pad_br(),
              dilation_, group_, algo_, iprop::forward_inference, X_dt, X.get_dims());
      if (filter_in.get_descriptor() != expected_descriptor) {
        // NOTE: MKL-DNN now only supports wino_reorder from (f32, any) to (s8, wino_fmt).
        // Thus, we first reorder weight from (s8, oihw) to (f32, oihw), then to (s8, wino_fmt).
        if (algo_ == ialgo::convolution_winograd && filter_in.get_data_type() != idtype::f32) {
          if (filter_f32_.is_empty()) {
            filter_f32_.init({filter_in.get_dims(), idtype::f32, filter_in.get_internal_format()});
            filter_f32_.feed_from(filter_in);
          }
          filter_in = filter_f32_;
        }

        filter_.init(expected_descriptor);
        filter_.set_scale(filter_scale);
        filter_.feed_from(filter_in);
      } else {
        filter_ = filter_in;
      }
 
      if (InputSize() > last_input_) {
        // NOTE: If the bias is shared by other operators in this module,
        // The existing bias scale should not satisfy current operator.
        // Thus, we have to requantize it by current input and filter scales.
        auto bias = Input(BIAS_OR_INPUT_S);
        bias_.init({bias.get_dims(), idtype::s32});
        iscale bias_scales (filter_.get_scale());
        //for (auto &scale : bias_scales) { scale *= X.get_scale()[0]; }
        if (X.get_scale().size() == bias_scales.size()) {
          for (int i = 0; i < bias_scales.size(); ++i) {
            bias_scales[i] *= X.get_scale()[i];
          }
        } else {
          for (auto &scale : bias_scales) { scale *= X.get_scale()[0]; }
        }

        bias_.set_scale(bias_scales);
        bias_.feed_from(bias);
      }
    }

    if (InputSize() > last_input_) {
      ideep::convolution_forward::compute(
          op_key_, comp, X_in, filter_, bias_, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_,
          iscale(), iscale(), Y_scales_, attr_, algo_,
          iprop::forward_inference, ipadding::zero, lowp_kind_);
    } else {
      ideep::convolution_forward::compute(
          op_key_, comp, X_in, filter_, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_,
          iscale(), iscale(), Y_scales_, attr_, algo_,
          iprop::forward_inference, ipadding::zero, lowp_kind_);
    }

    if (fusion_type_ != FUSION_CONV_RELU && fusion_type_ != FUSION_CONV_BRELU 
        && fusion_type_ != FUSION_UNKNOWN) {
      CAFFE_ENFORCE(
          Y == &(Input(InputSize() - 1)),
          "Convolution fusion op: InPlace is enforced for sum fusion.");
    }

    return true;
  }

 protected:
  iattr attr_;
  ialgo algo_;
  ikey op_key_;
  std::vector<float> scale_;
  int last_input_;
  int32_t zero_point_;
  ilowp_kind lowp_kind_;
  FusionType fusion_type_;
  ideep::convolution_forward::euler_params eparam;
  ideep::convolution_forward comp;
  ideep::format expect_x_format;
  itensor X_in, filter_, bias_, filter_f32_;
  iscale  Y_scales_;
  float X_scales_;
  std::vector<int> Y_dims;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;
  int need_euler_;
  INPUT_TAGS(INPUT_X, FILTER, BIAS_OR_INPUT_S, INPUT_S);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPInt8ConvReluOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    CAFFE_ENFORCE(zero_point_ == 0);
    last_input_ = BIAS_OR_INPUT_S;
    fusion_type_ = static_cast<FusionType>(
        OperatorBase::GetSingleArgument<int>("fusion_type", FUSION_CONV_RELU));
    CAFFE_ENFORCE(fusion_type_ == FUSION_CONV_RELU || fusion_type_ == FUSION_CONV_BRELU);
    if (fusion_type_ == FUSION_CONV_RELU) {
        attr_ = iattr::fuse_relu(); 
    } else {
      float bound_max = OperatorBase::GetSingleArgument<float>("bound_max", 0.0);
      float bound_min = OperatorBase::GetSingleArgument<float>("bound_min", 0.0);
      attr_ = iattr::fuse_relu(1.0, bound_max, bound_min, ialgo::eltwise_bounded_relu);
    }
  }
  virtual ~IDEEPInt8ConvReluOp() {}
};

class IDEEPInt8ConvSumOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    last_input_ = INPUT_S;
    attr_ = iattr::fuse_sum();
    fusion_type_ = FUSION_CONV_SUM;
  }
  virtual ~IDEEPInt8ConvSumOp() {}
};

class IDEEPInt8ConvSumReluOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvSumReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    last_input_ = INPUT_S;
    fusion_type_ = static_cast<FusionType>(
        OperatorBase::GetSingleArgument<int>("fusion_type", FUSION_CONV_SUM_RELU));
    CAFFE_ENFORCE(fusion_type_ == FUSION_CONV_SUM_RELU || fusion_type_ == FUSION_CONV_SUM_BRELU);
    if (fusion_type_ == FUSION_CONV_SUM_RELU) {
      attr_ = iattr::residual();
    } else if (fusion_type_ == FUSION_CONV_SUM_BRELU) {
      float sum_scale=1.0, relu_scale=1.0;
      float bound_max = OperatorBase::GetSingleArgument<float>("bound_max", 0.0);
      float bound_min = OperatorBase::GetSingleArgument<float>("bound_min", 0.0);
      attr_ = iattr::residual(sum_scale, relu_scale, bound_max, bound_min, ialgo::eltwise_bounded_relu);
    }
  }

  virtual ~IDEEPInt8ConvSumReluOp() {}
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Conv, DNNLOWP, IDEEPInt8ConvOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvRelu, DNNLOWP, IDEEPInt8ConvReluOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvSum, DNNLOWP, IDEEPInt8ConvSumOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvSumRelu, DNNLOWP, IDEEPInt8ConvSumReluOp);

OPERATOR_SCHEMA(Int8ConvSum)
    .NumInputs(2, 4)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .AllowInplace({{2, 0}, {3, 0}});

OPERATOR_SCHEMA(Int8ConvSumRelu)
    .NumInputs(2, 4)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .AllowInplace({{2, 0}, {3, 0}});

} // namespace caffe2

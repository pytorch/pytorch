#include "caffe2_dnnlowp_utils.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/sigmoid.h"
#include "caffe2/quantization/server/tanh.h"

#include <map>
#ifdef _OPENMP
#include <omp.h>
#endif

C10_DECLARE_int32(dnnlowp_activation_quantization_precision);
C10_DECLARE_int32(dnnlowp_weight_quantization_precision);
C10_DECLARE_int32(dnnlowp_requantization_multiplier_precision);
C10_DECLARE_int32(dnnlowp_eltwise_quantization_precision);
C10_DECLARE_bool(dnnlowp_force_scale_power_of_two);
C10_DECLARE_bool(dnnlowp_preserve_activation_sparsity);
C10_DECLARE_bool(dnnlowp_preserve_weight_sparsity);
C10_DECLARE_string(dnnlowp_activation_quantization_kind);
C10_DECLARE_string(dnnlowp_weight_quantization_kind);

namespace dnnlowp {

using namespace std;
using namespace caffe2;
using int8::Int8TensorCPU;

static bool HasDNNLowPEngine_(const OperatorDef& op_def) {
  const string ENGINE_PREFIX = "DNNLOWP";
  return strncmp(
             op_def.engine().c_str(),
             ENGINE_PREFIX.c_str(),
             ENGINE_PREFIX.size()) == 0;
}

static bool HasDNNLowPEngine_(const OperatorBase& op) {
  return HasDNNLowPEngine_(op.debug_def());
}

void PropagateOutputTensorQuantizationParams(
    OperatorBase *op, int idx, const TensorQuantizationParams& qparams) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
  Int8TensorCPU* output =
      op->Outputs()[idx]->template GetMutable<Int8TensorCPU>();
  output->scale = qparams.scale;
  output->zero_point = qparams.zero_point;
}

TensorQuantizationParams GetInputTensorQuantizationParamsOf(
    OperatorBase *op, int idx, const QuantizationFactory *qfactory,
    bool is_weight /*=false*/) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));

  if (op->InputIsType<Int8TensorCPU>(idx)) {
    const Int8TensorCPU& int8_tensor = op->Input<Int8TensorCPU>(idx);
    TensorQuantizationParams qparams;
    qparams.scale = int8_tensor.scale;
    qparams.zero_point = int8_tensor.zero_point;
    qparams.precision = qfactory->GetActivationPrecision();
    return qparams;
  }
  else {
    const TensorCPU *tensor = &op->template Input<Tensor>(idx, CPU);
    CAFFE_ENFORCE(tensor->template IsType<float>());
    CAFFE_ENFORCE(tensor->numel() == 0 || tensor->template data<float>());

    float min, max;
    FindMinMax(tensor->template data<float>(), &min, &max, tensor->numel());

    return qfactory->ChooseQuantizationParams(min, max, is_weight);
  }
}

static string OutputArgumentIdxString_(int idx) {
  return idx == 0 ? "" : to_string(idx + 1);
}

static string OutputScaleArgumentName(int idx) {
  return "Y" + OutputArgumentIdxString_(idx) + "_scale";
}

static string OutputZeroPointArgumentName(int idx) {
  return "Y" + OutputArgumentIdxString_(idx) + "_zero_point";
}

static void SetStaticQuantizationParams_(
    OperatorDef* op_def,
    int output_index,
    const TensorQuantizationParams& qparams) {
  AddArgument<float>(
      OutputScaleArgumentName(output_index), qparams.scale, op_def);
  AddArgument<int32_t>(
      OutputZeroPointArgumentName(output_index), qparams.zero_point, op_def);
}

void SetStaticQuantizationParams(
    OperatorBase* op,
    int output_index,
    const TensorQuantizationParams& qparams) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
  auto op_def = make_shared<OperatorDef>();
  *op_def = op->debug_def();
  SetStaticQuantizationParams_(op_def.get(), output_index, qparams);
  op->set_debug_def(op_def);
}

bool HasStaticQuantization(
    const caffe2::OperatorBase* op, int output_index /*=0*/) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
  return op->HasSingleArgumentOfType<float>(
      OutputScaleArgumentName(output_index));
}

TensorQuantizationParams GetStaticQuantizationParamsOf(
    const caffe2::OperatorBase* op, int idx) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
  unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);

  TensorQuantizationParams qparams;
  qparams.scale = op->GetSingleArgument<float>(OutputScaleArgumentName(idx), 0);
  qparams.zero_point =
      op->GetSingleArgument<int32_t>(OutputZeroPointArgumentName(idx), 0);
  qparams.precision = qfactory->GetActivationPrecision();

  return qparams;
}

template <typename T>
const T* QuantizeInputIfNeeded(
  OperatorBase* op, int input_index,
  const TensorQuantizationParams& qparams, vector<T>& temp,
  const QuantizationFactory *qfactory) {
  if (op->InputIsType<int8::Int8TensorCPU>(input_index)) {
    // Already quantized
    return op->Input<int8::Int8TensorCPU>(input_index).t.data<T>();
  } else {
    // Need to quantize
    const TensorCPU& tensor = op->Input<Tensor>(input_index, CPU);
    temp.resize(tensor.numel());
    Quantize<T>(tensor.data<float>(), temp.data(), temp.size(), qparams);
    return temp.data();
  }
}

template <typename T>
const T* RowWiseQuantizeInputIfNeeded(
  OperatorBase* op, int input_index,
  const std::vector<TensorQuantizationParams>& qparams, vector<T>& temp,
  const QuantizationFactory *qfactory) {
  if (op->InputIsType<int8::Int8TensorCPU>(input_index)) {
    // Already quantized
    return op->Input<int8::Int8TensorCPU>(input_index).t.data<T>();
  } else {
    // Need to quantize
    const TensorCPU& tensor = op->Input<Tensor>(input_index, CPU);
    temp.resize(tensor.numel());
    // number of rows
    int N = qparams.size();
    int rowwidth = temp.size()/N;
    // quantize each row
    for (int i = 0; i < N; i++) {
      Quantize<T>(
          tensor.data<float>() + rowwidth * i,
          temp.data() + rowwidth * i,
          rowwidth,
          qparams[i]);
    }
    return temp.data();
  }
}

template
const uint8_t *QuantizeInputIfNeeded<uint8_t>(
  OperatorBase *op, int input_index,
  const TensorQuantizationParams& qparams, vector<uint8_t>& temp,
  const QuantizationFactory *qfactory);

template
const int8_t *QuantizeInputIfNeeded<int8_t>(
  OperatorBase *op, int input_index,
  const TensorQuantizationParams& qparams, vector<int8_t>& temp,
  const QuantizationFactory *qfactory);

template
const uint16_t *QuantizeInputIfNeeded<uint16_t>(
  OperatorBase *op, int input_index,
  const TensorQuantizationParams& qparams, vector<uint16_t>& temp,
  const QuantizationFactory *qfactory);

template
const int16_t *QuantizeInputIfNeeded<int16_t>(
  OperatorBase *op, int input_index,
  const TensorQuantizationParams& qparams, vector<int16_t>& temp,
  const QuantizationFactory *qfactory);

template
const uint8_t *RowWiseQuantizeInputIfNeeded<uint8_t>(
  OperatorBase *op, int input_index,
  const std::vector<TensorQuantizationParams>& qparams, vector<uint8_t>& temp,
  const QuantizationFactory *qfactory);

template
const uint16_t *RowWiseQuantizeInputIfNeeded<uint16_t>(
  OperatorBase *op, int input_index,
  const std::vector<TensorQuantizationParams>& qparams, vector<uint16_t>& temp,
  const QuantizationFactory *qfactory);

void MeasureQuantizationError(
    const float* actual,
    const float* ref,
    size_t len,
    QuantizationErrorStats* stat) {
  for (int i = 0; i < len; ++i) {
    stat->sum_sq += ref[i] * ref[i];
    float err = actual[i] - ref[i];
    stat->sum_err_sq += err * err;

    if (fabs(err) > stat->max_abs_err) {
      stat->max_abs_err = fabs(err);
      stat->max_err_actual = actual[i];
      stat->max_err_ref = ref[i];
    }
  }
  ++stat->measure_cnt;
}

void ReportQuantizationError(
    const OperatorBase* op,
    const QuantizationErrorStats& stat) {
  if (stat.sum_sq == 0) {
    LOG(INFO) << " output " << op->debug_def().output(0) << " of operator "
              << op << " with type " << op->debug_def().type()
              << " has l2 relative error nan (stat.sum_err_sq "
              << stat.sum_err_sq << " stat.sum_sq 0)"
              << " and max abs error " << stat.max_abs_err << " (reference is "
              << stat.max_err_ref << " and actual is " << stat.max_err_actual
              << ")"
              << " sum_err_sq " << stat.sum_err_sq << " sum_sq_ " << stat.sum_sq
              << " cnt " << stat.measure_cnt;
  } else {
    LOG(INFO) << " output " << op->debug_def().output(0) << " of operator "
              << op << " with type " << op->debug_def().type()
              << " has l2 relative error "
              << std::sqrt(stat.sum_err_sq) / std::sqrt(stat.sum_sq)
              << " and max abs error " << stat.max_abs_err << " (reference is "
              << stat.max_err_ref << " and actual is " << stat.max_err_actual
              << ")"
              << " sum_err_sq " << stat.sum_err_sq << " sum_sq_ " << stat.sum_sq
              << " cnt " << stat.measure_cnt;
  }
}

static unique_ptr<QuantizationFactory> GetQuantizationFactoryOf_(
    const OperatorDef& op_def) {
  int activation_precision =
      ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          op_def, "activation_precision",
          FLAGS_dnnlowp_activation_quantization_precision);
  int weight_precision = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
      op_def, "weight_precision", FLAGS_dnnlowp_weight_quantization_precision);
  int requantization_multiplier_precision =
      ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          op_def,
          "requantization_multiplier_precision",
          FLAGS_dnnlowp_requantization_multiplier_precision);
  int eltwise_quantization_precision =
      ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          op_def,
          "eltwise_quantization_precision",
          FLAGS_dnnlowp_eltwise_quantization_precision);
  bool preserve_activation_sparsity =
      ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
          op_def,
          "preserve_activation_sparsity",
          FLAGS_dnnlowp_preserve_activation_sparsity);
  bool preserve_weight_sparsity =
      ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
          op_def, "preserve_weight_sparsity",
          FLAGS_dnnlowp_preserve_weight_sparsity);
  bool force_scale_power_of_two =
      ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
          op_def, "force_scale_power_of_two",
          FLAGS_dnnlowp_force_scale_power_of_two);
  string activation_quantization_kind =
      ArgumentHelper::GetSingleArgument<OperatorDef, string>(
          op_def,
          "activation_quantization_kind",
          FLAGS_dnnlowp_activation_quantization_kind);
  string weight_quantization_kind =
      ArgumentHelper::GetSingleArgument<OperatorDef, string>(
          op_def, "weight_quantization_kind",
          FLAGS_dnnlowp_weight_quantization_kind);

  VLOG(2) << "Quantization method for op with output " << op_def.output(0)
          << " activation_precision " << activation_precision
          << " weight_precision " << weight_precision
          << " requantization_multiplier_precision "
          << requantization_multiplier_precision
          << " eltwise_quantization_precision "
          << eltwise_quantization_precision << " preserve_activation_sparsity "
          << preserve_activation_sparsity << " preserve_weight_sparsity "
          << preserve_weight_sparsity << " force_scale_power_of_two "
          << force_scale_power_of_two << " activation_quantization_kind "
          << activation_quantization_kind << " weight_quantization_kind "
          << weight_quantization_kind;

  return unique_ptr<QuantizationFactory>(
    new QuantizationFactory(
      activation_precision,
      weight_precision,
      requantization_multiplier_precision,
      eltwise_quantization_precision,
      preserve_activation_sparsity,
      preserve_weight_sparsity,
      force_scale_power_of_two,
      StringToKind(activation_quantization_kind),
      StringToKind(weight_quantization_kind)));
}

unique_ptr<QuantizationFactory> GetQuantizationFactoryOf(
    const OperatorBase* op) {
  return GetQuantizationFactoryOf_(op->debug_def());
}

void AdjustOutputTensorQuantizationParamsWithFollowedBy(
    OperatorBase* op,
    const string& followed_by) {
  LOG_IF(WARNING, !HasDNNLowPEngine_(*op));

  auto op_def = make_shared<OperatorDef>();
  *op_def = op->debug_def();
  AddArgument<string>("followed_by", followed_by, op_def.get());
  op->set_debug_def(op_def);

  if (followed_by == "Sigmoid") {
    SetStaticQuantizationParams(
        op, 0, Sigmoid<uint8_t>().GetInputQuantizationParams());
  } else if (followed_by == "Tanh") {
    SetStaticQuantizationParams(
        op, 0, Tanh<uint8_t>().GetInputQuantizationParams());
  } else if (followed_by == "Relu") {
    if (HasStaticQuantization(op)) {
      unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);
      TensorQuantizationParams qparams = GetStaticQuantizationParamsOf(op, 0);
      qparams = qfactory->ChooseQuantizationParams(0, qparams.Max());
      SetStaticQuantizationParams(op, 0, qparams);
    }
  } else {
    LOG(WARNING) << "Unknown followed_by " << followed_by;
  }
}

void ParseDNNLowPOperatorArguments(
    OperatorBase* op,
    bool* dequantize_output,
    bool* measure_quantization_error,
    string* followed_by) {

  // When exiting quantized region or we're just doing per-op quantization,
  // dequantize the outputs as floats.
  if (dequantize_output) {
    *dequantize_output =
      op->GetSingleArgument<bool>("dequantize_output", false);
    if (*dequantize_output) {
      VLOG(2) << "Dequantize output " << op->debug_def().output(0)
              << " of operator type " << op->debug_def().type();
    }
  }

  // Measure quantization error by comparing with reference fp32 operators.
  if (measure_quantization_error) {
    *measure_quantization_error =
      op->GetSingleArgument<bool>("measure_quantization_error", false);
  }

  // Output scale and zero_point can be specified (actually recommended to be
  // specified for performance to avoid on-the-fly quantization parameter
  // selection) from activation distributions collected from profiling.
  if (HasStaticQuantization(op)) {
    TensorQuantizationParams qparams = GetStaticQuantizationParamsOf(op, 0);
    unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);
    if (qparams.zero_point != (1 << (qfactory->GetActivationPrecision() - 1)) &&
        qparams.zero_point != 0 &&
        qfactory->GetPreserveActivationSparsity()) {
      LOG(WARNING) << "Symmetric quantization is used for activation but "
                      "Y_zero_point is " << qparams.zero_point << " for "
                   << op->debug_def().output(0)
                   << " output activation of an operator with type "
                   << op->debug_def().type();
    }
  } else {
    if (op->HasSingleArgumentOfType<int>("Y_zero_point")) {
      LOG(WARNING) << "Y_zero_point without Y_scale for "
                   << op->debug_def().output(0)
                   << " (an output of operator type "
                   << op->debug_def().type() << ") doesn't make sense";
    }
  }

  // When an operator has only one consumer and the consumer only cares about
  // a limited range of values, we can quantize more precisely.
  if (op->HasSingleArgumentOfType<string>("followed_by")) {
    string followed_by_ =
      op->GetSingleArgument<string>("followed_by", "");
    VLOG(2) << "Operator with type " << op->debug_def().type()
            << " and output " << op->debug_def().output(0)
            << " is followed by " << followed_by_;

    AdjustOutputTensorQuantizationParamsWithFollowedBy(op, followed_by_);
    if (followed_by) {
      *followed_by = followed_by_;
    }
  }
}

NetDef AddScaleZeroOffsetArgumentsWithHistogram(
    NetDef net_def, const string& histogram_file_name) {
  ifstream f(histogram_file_name);

  // check the format by looking at the first line
  string first_line, word;
  getline(f, first_line);
  f.seekg(0, f.beg);
  istringstream ist(first_line);
  int nwords_first_line = 0;
  while (ist >> word) {
    ++nwords_first_line;
  }

  ist = istringstream(first_line);

  bool new_format = true;
  int op_index, i, nbins;
  string op_type, tensor_name;
  float min, max;
  ist >> op_index >> op_type >> i >> tensor_name >> min >> max >> nbins;
  if (nwords_first_line != nbins + 7) {
    ist = istringstream(first_line);
    ist >> op_index >> i >> tensor_name >> min >> max >> nbins;
    if (nwords_first_line == nbins + 6) {
      new_format = false;
    }
    else {
      LOG(WARNING)
        << "histogram file " << histogram_file_name << " has an invalid format";
      return net_def;
    }
  }

  // parse the input file
  op_index = 0;
  for (auto& op_def : *net_def.mutable_op()) {
    ArgumentHelper arg_helper(op_def);

    for (i = 0; i < op_def.output().size(); ++i) {
      int op_index2, i2;

      if (new_format) {
        f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max >> nbins;
      }
      else {
        f >> op_index2 >> i2 >> tensor_name >> min >> max >> nbins;
      }
      LOG_IF(WARNING, op_index2 != op_index) <<
        "op index " << op_index2 << " doesn't match with " << op_index;
      LOG_IF(WARNING, tensor_name != op_def.output(i)) <<
        tensor_name << " in histogram file line " << op_index <<
        " doesn't match with operation def " <<
        op_def.output(i);
      LOG_IF(WARNING, i2 != i) <<
        "output tensor index " << i2 << " doesn't match with " << i;
      if (new_format) {
        LOG_IF(WARNING, op_type != op_def.type()) <<
          "operator type " << op_type << " in histogram file line " <<
          op_index << " doesn't match with operation def " <<
          op_def.type();
      }

      vector<uint64_t> bins;
      for (int j = 0; j < nbins; ++j) {
        uint64_t cnt;
        f >> cnt;
        bins.push_back(cnt);
      }

      if (!HasDNNLowPEngine_(op_def) ||
          arg_helper.GetSingleArgument<int>("dequantize_output", 0) != 0 ||
          i > 0) {
        LOG(INFO) << "Skip " << op_def.type() << " " << op_def.output(0);
        continue;
      }

      Histogram hist = Histogram(min, max, bins);

      unique_ptr<QuantizationFactory> qfactory =
          GetQuantizationFactoryOf_(op_def);
      TensorQuantizationParams qparams =
          qfactory->ChooseQuantizationParams(hist);

      SetStaticQuantizationParams_(&op_def, 0, qparams);
    }
    ++op_index;
  }

  return net_def;
}

} // namespace dnnlowp

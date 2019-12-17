#include "caffe2/core/operator_schema.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

bool OpSchema::Verify(const OperatorDef& def) const {
  // Check the number of inputs.
  if (def.input_size() < min_input_ || def.input_size() > max_input_) {
    LOG(ERROR) << "Input size " << def.input_size()
                    << " not in range [min=" << min_input_ << ", max="
                    << max_input_ << "].";
    return false;
  }
  if (!num_inputs_allowed_(def.input_size())) {
    LOG(ERROR) << "Input size " << def.input_size()
                    << " not in allowed input sizes.";
    return false;
  }
  // Check the number of outputs.
  if (def.output_size() < min_output_ || def.output_size() > max_output_) {
    LOG(ERROR) << "Output size " << def.output_size()
                    << " not in range [min=" << min_output_ << ", max="
                    << max_output_ << "].";
    return false;
  }
  if (!num_outputs_allowed_(def.output_size())) {
    LOG(ERROR) << "Output size " << def.output_size()
                    << " not in allowed output sizes.";
    return false;
  }
  if (!num_inputs_outputs_allowed_(def.input_size(), def.output_size())) {
    LOG(ERROR) << "Combination of input size " << def.input_size()
               << "and output size " << def.output_size() << " not in allowed.";
    return false;
  }
  // If the number of outputs can be calculated, check if the number matches.
  if (calculate_output_) {
    int expected_nout = calculate_output_(def.input_size());
    if (expected_nout != kCannotComputeNumOutputs &&
        def.output_size() != expected_nout) {
      LOG(ERROR) << "Output size " << def.output_size()
                      << " not matching expected output size, which is "
                      << expected_nout;
      return false;
    }
  }

  // Check in-place settings.
  for (int in_idx = 0; in_idx < def.input_size(); ++in_idx) {
    for (int out_idx = 0; out_idx < def.output_size(); ++out_idx) {
      // If an input is the same as an output but in-place is not opt-in
      // either as allowed or enforced, we will fail the verification.
      if (def.input(in_idx) == def.output(out_idx) &&
          (!inplace_allowed_(in_idx, out_idx)
          && !inplace_enforced_(in_idx, out_idx))) {
        LOG(ERROR) << "Input index " << in_idx << " and output idx " << out_idx
                   << " (" << def.input(in_idx) << ")"
                   << " are set to be in-place but this is actually not "
                   << "supported by op " << def.type();
        return false;
      }
      if (def.input(in_idx) != def.output(out_idx) &&
          inplace_enforced_(in_idx, out_idx)) {
        LOG(ERROR) << "Input index " << in_idx << " (" << def.input(in_idx) << ")"
                   << " and output idx " << out_idx
                   << " (" << def.output(in_idx) << ")"
                   << " are not in-place but should be as required by op "
                   << def.type();
        return false;
      }
    }
  }

  std::set<std::string> present_args{};
  for (const auto& arg : def.arg()) {
    present_args.insert(arg.name());
  }

  for (const auto& arg : args()) {
    if (arg.is_required() &&
        present_args.find(arg.name()) == present_args.end()) {
      LOG(ERROR) << "Argument '" << arg.name() << "' is required for Operator '"
                 << def.type() << "'.";
      return false;
    }
  }

  // Phew. All verifications passed.
  return true;
}

OpSchema& OpSchema::NumInputs(int min, int max) {
  min_input_ = min;
  max_input_ = max;
  return *this;
}

OpSchema& OpSchema::NumInputs(int n) {
  return NumInputs(n, n);
}

OpSchema& OpSchema::NumInputs(std::function<bool(int)> func) {
  num_inputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumInputs(set<int> allowed_input_nums) {
  return NumInputs(
      [allowed_input_nums](int n)->bool {
        return allowed_input_nums.count(n);
      });
}

OpSchema& OpSchema::NumOutputs(int min, int max) {
  min_output_ = min;
  max_output_ = max;
  return *this;
}

OpSchema& OpSchema::NumOutputs(int n) {
  return NumOutputs(n, n);
}

OpSchema& OpSchema::NumOutputs(std::function<bool(int)> func) {
  num_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumOutputs(set<int> allowed_output_nums) {
  return NumOutputs(
      [allowed_output_nums](int n)->bool {
        return allowed_output_nums.count(n);
      });
}

OpSchema& OpSchema::NumInputsOutputs(std::function<bool(int, int)> func) {
  num_inputs_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::OutputCalculator(std::function<int(int)> calc) {
  calculate_output_ = calc;
  return *this;
}

OpSchema& OpSchema::SameNumberOfOutput() {
  return OutputCalculator([](int n)->int { return n; } );
}

OpSchema& OpSchema::AllowInplace(std::function<bool(int, int)> inplace) {
  inplace_allowed_ = inplace;
  return *this;
}

OpSchema& OpSchema::AllowInplace(set<std::pair<int, int>> inplace) {
  return AllowInplace(
      [inplace](int in, int out)->bool {
        return inplace.count(std::make_pair(in, out));
      });
}

OpSchema& OpSchema::AllowOneToOneInplace() {
  return AllowInplace([](int in, int out) { return in == out; });
}

OpSchema& OpSchema::EnforceInplace(std::function<bool(int, int)> inplace) {
  inplace_enforced_ = inplace;
  return *this;
}

OpSchema& OpSchema::EnforceInplace(set<std::pair<int, int>> inplace) {
  return EnforceInplace(
      [inplace](int in, int out)->bool {
        return inplace.count(std::make_pair(in, out));
      });
}

OpSchema& OpSchema::EnforceOneToOneInplace() {
  return EnforceInplace([](int in, int out) { return in == out; });
}

OpSchema& OpSchema::Private() {
  private_ = true;
  return *this;
}

OpSchema& OpSchema::InputsCanCrossDevices() {
  inputs_can_cross_devices_ = true;
  return *this;
}

OpSchema& OpSchema::TensorInferenceFunction(
    TensorInferenceFunctionType function) {
  tensor_inference_function_ = function;
  return *this;
}

OpSchema::TensorInferenceFunctionType OpSchema::NeedsAllInputShapes(
    TensorInferenceFunctionType f) {
  return [f](const OperatorDef& def, const vector<TensorShape>& in) {
    for (const auto& in_ts : in) {
      if (in_ts.unknown_shape()) {
        vector<TensorShape> out(def.output().size());
        for (auto& out_ts : out) {
          out_ts.set_unknown_shape(true);
        }
        return out;
      }
    }
    return f(def, in);
  };
}

OpSchema& OpSchema::InheritOnnxSchema(const std::string& onnx_schema_name) {
  onnx_schema_ = onnx_schema_name;
  return *this;
}

OpSchema& OpSchema::IdenticalTypeAndShape() {
  return TensorInferenceFunction(
      [](const OperatorDef&, const vector<TensorShape>& input_types) {
        return vector<TensorShape>(input_types);
      });
}

OpSchema& OpSchema::IdenticalTypeAndShapeOfInput(int idx) {
  return TensorInferenceFunction(
      [idx](const OperatorDef&, const vector<TensorShape>& input_types) {
        vector<TensorShape> out(1);
        out[0] = input_types[idx];
        return out;
      });
}

OpSchema& OpSchema::IdenticalTypeAndShapeOfMultipleInputs(
    const vector<int>& indices) {
  return TensorInferenceFunction(
      [indices](const OperatorDef&, const vector<TensorShape>& input_types) {
        vector<TensorShape> out(indices.size());
        for (int i = 0; i < indices.size(); i++) {
          out[i] = input_types[indices.at(i)];
        }
        return out;
      });
}

OpSchema& OpSchema::IdenticalTypeAndShapeOfInputDim(int idx, int dim) {
  return TensorInferenceFunction(
      [idx, dim](const OperatorDef&, const vector<TensorShape>& input_types) {
        vector<TensorShape> out(1);
        out[0].add_dims(input_types[idx].dims(dim));
        out[0].set_data_type(input_types[idx].data_type());
        return out;
      });
}

OpSchema& OpSchema::ScalarType(::caffe2::TensorProto_DataType dt) {
  return TensorInferenceFunction(
      [dt](const OperatorDef& def, const vector<TensorShape>& /*input_types*/) {
        TensorShape shape;
        shape.set_data_type(dt);
        vector<TensorShape> out(def.output_size(), shape);
        return out;
      });
}

OpSchema& OpSchema::CostInferenceFunction(CostInferenceFunctionType function) {
  cost_inference_function_ =
      std::make_unique<CostInferenceFunctionType>(function);
  return *this;
}

OpSchema& OpSchema::DeviceInferenceFunction(
    DeviceInferenceFunctionType function) {
  device_inference_function_ = function;
  return *this;
}

OpSchema& OpSchema::SetDoc(const string& doc) {
  doc_ = doc;
  return *this;
}

OpSchema&
OpSchema::Arg(const char* name, const char* description, bool required) {
  args_.push_back(Argument(name, description, required));
  return *this;
}

#define DEFINE_STANDARG_ARG(name, str)                                \
  CAFFE2_API const char* OpSchema::Arg_##name = #str;                 \
  CAFFE2_API OpSchema& OpSchema::Arg##name(const char* description) { \
    return Arg(#str, description, true);                              \
  }

DEFINE_STANDARG_ARG(IsTest, is_test)

#undef DEFINE_STANDARG_ARG

OpSchema& OpSchema::Input(const int n, const char* name, const char* description) {
  if (input_desc_.size() <= (unsigned)n) {
    input_desc_.resize(n + 1);
  }
  input_desc_[n] = std::make_pair(name, description);
  return *this;
}

OpSchema& OpSchema::Output(const int n, const char* name, const char* description) {
  if (output_desc_.size() <= (unsigned)n) {
    output_desc_.resize(n + 1);
  }
  output_desc_[n] = std::make_pair(name, description);
  return *this;
}

OpSchema& OpSchema::FillUsing(std::function<void(OpSchema&)> populator) {
  if (populator) {
    populator(*this);
  }
  return *this;
}

int OpSchema::CalculateOutput(int num_input) const {
  if (min_output_ == max_output_) {
    return min_output_;
  } else if (calculate_output_) {
    return calculate_output_(num_input);
  } else {
    return kCannotComputeNumOutputs;
  }
}

namespace {
void SparseLengthsFillerHelper(
    const std::vector<std::vector<int64_t>>& shapes,
    size_t value_index,
    size_t length_index,
    std::vector<TensorFiller>* fillers) {
  CAFFE_ENFORCE_EQ(shapes[length_index].size(), 1);
  // filler.h: SparseLengths->FixedSum will select FD_FIXEDSUM distribution
  (*fillers)[length_index].SparseLengths(shapes[value_index].front());
}

void SparseWeightsFillerHelper(
    const std::vector<std::vector<int64_t>>& shapes,
    size_t weight_index,
    std::vector<TensorFiller>* fillers) {
  (*fillers)[weight_index]
      .Min(0)
      .Max(shapes[weight_index].front())
      .Dist(FD_UNIFORM);
}

void SparseSegmentsFillerHelper(
    const std::vector<std::vector<int64_t>>& shapes,
    size_t value_index,
    size_t segment_index,
    std::vector<TensorFiller>* fillers) {
  CAFFE_ENFORCE_EQ(shapes[segment_index].size(), 1);
  // filler.h SparseSegments will select FD_UNIFORM or FD_SYNTHETIC distribution
  (*fillers)[value_index]
      .Min(0)
      .Max(shapes[value_index].front() * 2)
      .Dist(FD_UNIFORM);
  (*fillers)[segment_index].SparseSegments(shapes[value_index].front() - 1);
}
} // namespace

// The helper is build sparse input with values, keys, and lengths; e.g.:
// values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
// keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
//            \_____/  \________/  \__/
// lengths =    [3,        4,       2]
OpSchema& OpSchema::ValueKeyLengthInputFillers(
    size_t value_index,
    size_t key_index,
    size_t length_index) {
  filler_supplier_ = [this, value_index, key_index, length_index](
                         const std::vector<std::vector<int64_t>>& shapes) {
    auto fillers = SupplyDenseFillers(shapes);
    // fill in the length (value_index is used to get the correct shape)
    SparseLengthsFillerHelper(shapes, key_index, length_index, &fillers);
    // fill in the keys (value_index is used to get the correct shape)
    SparseSegmentsFillerHelper(shapes, value_index, key_index, &fillers);
    return fillers;
  };
  return *this;
}

// The helper is build sparse input with values, keys, and lengths; e.g.:
// values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
// keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
// weights = [1, 1, 1, 0, 2, 2, 2, 1, 2]
//            \_____/  \________/  \__/
// lengths =    [3,        4,       2]
OpSchema& OpSchema::WeightedValueKeyLengthInputFillers(
    size_t value_index,
    size_t key_index,
    size_t length_index,
    size_t weight_index) {
  filler_supplier_ = [this, value_index, key_index, length_index, weight_index](
                         const std::vector<std::vector<int64_t>>& shapes) {
    auto fillers = SupplyDenseFillers(shapes);
    // fill in the length (value_index is used to get the correct shape)
    SparseLengthsFillerHelper(shapes, key_index, length_index, &fillers);
    // fill in the keys (value_index is used to get the correct shape)
    SparseSegmentsFillerHelper(shapes, value_index, key_index, &fillers);
    // fill in the weights
    SparseWeightsFillerHelper(shapes, weight_index, &fillers);
    return fillers;
  };
  return *this;
}

// The helper is build sparse input with values and lengths; e.g.:
// values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
//            \_____/  \________/  \__/
// lengths =    [3,        4,       2]
OpSchema& OpSchema::ValueLengthInputFillers(
    size_t value_index,
    size_t length_index) {
  filler_supplier_ = [this, value_index, length_index](
                         const std::vector<std::vector<int64_t>>& shapes) {
    auto fillers = SupplyDenseFillers(shapes);
    // fill in the length (value_index is used to get the correct shape)
    SparseLengthsFillerHelper(shapes, value_index, length_index, &fillers);
    return fillers;
  };
  return *this;
}

OpSchema& OpSchema::DisallowInputFillers() {
  filler_supplier_ =
      [this](const std::vector<std::vector<int64_t>>& /* unused */) {
        throw std::invalid_argument(type_ + " does not have input fillers");
        return std::vector<TensorFiller>();
      };
  return *this;
}

std::vector<TensorFiller> OpSchema::InputFillers(
    const std::vector<std::vector<int64_t>>& shapes) const {
  return filler_supplier_(shapes);
}

std::vector<TensorFiller> OpSchema::SupplyDenseFillers(
    const std::vector<std::vector<int64_t>>& shapes) {
  std::vector<TensorFiller> fillers;
  for (const auto& shape : shapes) {
    fillers.emplace_back(shape);
  }
  return fillers;
}

C10_EXPORT std::ostream& operator<<(std::ostream& out, const OpSchema& schema) {
  if (!schema.args().empty()) {
    out << "Arguments:" << std::endl;
    for (const auto& arg : schema.args()) {
      out << "  " << arg.name() << " : " << arg.description() << std::endl;
    }
  }
  if (schema.max_input_ > 0) {
    out << "Inputs:" << std::endl;
    if (!schema.input_desc_.empty()) {
      for (size_t i = 0; i < schema.input_desc_.size(); ++i) {
        const auto& p = schema.input_desc_[i];
        out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
            << (p.second ? p.second : "(no doc)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  if (schema.max_output_ > 0) {
    out << "Outputs:" << std::endl;
    if (!schema.output_desc_.empty()) {
      for (size_t i = 0; i < schema.output_desc_.size(); ++i) {
        const auto& p = schema.output_desc_[i];
        out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
            << (p.second ? p.second : "(no doc)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  out << std::endl;
  if (schema.doc()) {
    out << schema.doc();
  } else {
    out << "(no documentation yet)" << std::endl;
  }
  out << std::endl;
  if (schema.line_) {
    out << "Defined at " << schema.file_ << ":" << schema.line_ << std::endl;
  }
  return out;
}

CaffeMap<string, OpSchema>& OpSchemaRegistry::map() {
  static CaffeMap<string, OpSchema> map;
  return map;
}

}  // namespace caffe2

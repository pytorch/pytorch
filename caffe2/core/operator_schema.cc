#include "caffe2/core/operator_schema.h"

#include "caffe2/core/logging.h"

namespace caffe2 {

bool OpSchema::Verify(const OperatorDef& def) const {
  // Check the number of inputs.
  if (def.input_size() < min_input_ || def.input_size() > max_input_) {
    CAFFE_LOG_ERROR << "Input size " << def.input_size()
                    << " not in range [min=" << min_input_ << ", max="
                    << max_input_ << "].";
    return false;
  }
  if (!num_inputs_allowed_(def.input_size())) {
    CAFFE_LOG_ERROR << "Input size " << def.input_size()
                    << " not in allowed input sizes.";
    return false;
  }
  // Check the number of outputs.
  if (def.output_size() < min_output_ || def.output_size() > max_output_) {
    CAFFE_LOG_ERROR << "Output size " << def.output_size()
                    << " not in range [min=" << min_output_ << ", max="
                    << max_output_ << "].";
    return false;
  }
  if (!num_outputs_allowed_(def.output_size())) {
    CAFFE_LOG_ERROR << "Output size " << def.output_size()
                    << " not in allowed output sizes.";
    return false;
  }
  // If the number of outputs can be calculated, check if the number matches.
  if (calculate_output_) {
    int expected_nout = calculate_output_(def.input_size());
    if (expected_nout != kCannotComputeNumOutputs &&
        def.output_size() != expected_nout) {
      CAFFE_LOG_ERROR << "Output size " << def.output_size()
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
        CAFFE_LOG_ERROR
            << "Input index " << in_idx << " and output idx "
            << out_idx << " are set to be in-place but this is actually not "
            << "supported by op " << def.type();
        return false;
      }
      if (def.input(in_idx) != def.output(out_idx) &&
          inplace_enforced_(in_idx, out_idx)) {
        CAFFE_LOG_ERROR
            << "Input index " << in_idx << " and output idx " << out_idx
            << " are not in-place but should be as required by op "
            << def.type();
        return false;
      }
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

int OpSchema::CalculateOutput(int num_input) const {
  if (min_output_ == max_output_) {
    return min_output_;
  } else if (calculate_output_) {
    return calculate_output_(num_input);
  } else {
    return kCannotComputeNumOutputs;
  }
}


CaffeMap<string, OpSchema>& OpSchemaRegistry::map() {
  static CaffeMap<string, OpSchema> map;
  return map;
}

}  // namespace caffe2

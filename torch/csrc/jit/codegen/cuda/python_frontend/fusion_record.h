#pragma once
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>

namespace nvfuser {
struct RecordFunctor {
  RecordFunctor(std::vector<size_t> _args, std::vector<size_t> _outputs) :
    args(std::move(_args)),
    outputs(std::move(_outputs)) {}
  virtual void operator()(FusionDefinition& fd) = 0;

  std::vector<size_t> args;
  std::vector<size_t> outputs;
};

struct InputTensorRecord : RecordFunctor {
  InputTensorRecord(std::vector<size_t> _outputs, 
                    std::vector<int64_t> _symbolic_sizes,
                    std::vector<bool> _contiguous_info,
                    NvfDataType _dtype):
    RecordFunctor({}, std::move(_outputs)),
    symbolic_sizes(std::move(_symbolic_sizes)),
    contiguous_info(std::move(_contiguous_info)),
    dtype(_dtype) {}
  void operator()(FusionDefinition &fd) final {
    auto tv = TensorViewBuilder()
                  .ndims(symbolic_sizes.size())
                  .contiguity(contiguous_info)
                  .shape(symbolic_sizes)
                  .dtype(dtype)
                  .build();
    
    fd.fusion_state[outputs[0]] = tv;
    fd.addInput(tv);
  }

  std::vector<int64_t> symbolic_sizes;
  std::vector<bool> contiguous_info;
  NvfDataType dtype;
};

template<class OutputType>
struct OutputRecord : RecordFunctor {
  OutputRecord(std::vector<size_t> _args):
    RecordFunctor(std::move(_args), {}) {}

  void operator()(FusionDefinition &fd) final {
    auto input = fd.fusion_state[args[0]];

    // With C++17, this statement should be "if constexpr" 
    if (std::is_same<OutputType, NvfTensorView>::value) {
      fd.addOutput(input->as<NvfTensorView>());
    } else {
      fd.addOutput(input);
    }
  }
};

// With C++17, this template specialization could be replaced by
// "if constexpr" statement in the UnaryOpRecord::operator() 
// function.
template<class T>
T* unary_op_func(std::function<T*(T*)> func, NvfVal* arg) {
  return func(arg);
}
template<>
NvfTensorView* unary_op_func<NvfTensorView>(
    std::function<NvfTensorView*(NvfTensorView*)> func, NvfVal* arg) {
  return func(arg->as<NvfTensorView>());
}

template<class ArgType>
struct UnaryOpRecord : RecordFunctor {
  UnaryOpRecord(std::vector<size_t> _args,
                std::vector<size_t> _outputs,
                std::function<ArgType*(ArgType*)> fusion_op) :
    RecordFunctor(std::move(_args), std::move(_outputs)),
    fusion_op_(fusion_op) {}

  void operator()(FusionDefinition& fd) final {
    auto input = fd.fusion_state[args[0]];
    auto output = unary_op_func<ArgType>(fusion_op_, input);
    fd.fusion_state[outputs[0]] = output;
  }

 private:
  std::function<ArgType*(ArgType*)> fusion_op_;
};

} // nvfuser namespace

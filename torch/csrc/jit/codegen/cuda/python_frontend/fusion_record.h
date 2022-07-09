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

// With C++17, this template specialization could be replaced by
// "if constexpr" statement in the BinaryOpRecord::operator() 
// function.
template<class O, class A1, class A2>
O* binary_op_func(std::function<O*(A1*, A2*)> func, NvfVal* arg1, NvfVal* arg2) {
  return func(arg1, arg2);
}
template<>
NvfTensorView* binary_op_func<NvfTensorView, NvfTensorView, NvfTensorView>(
    std::function<NvfTensorView*(NvfTensorView*, NvfTensorView*)> func, NvfVal* arg1, NvfVal* arg2) {
  return func(arg1->as<NvfTensorView>(), arg2->as<NvfTensorView>());
}
template<>
NvfTensorView* binary_op_func<NvfTensorView, NvfTensorView, NvfVal>(
    std::function<NvfTensorView*(NvfTensorView*, NvfVal*)> func, NvfVal* arg1, NvfVal* arg2) {
  return func(arg1->as<NvfTensorView>(), arg2);
}
template<>
NvfTensorView* binary_op_func<NvfTensorView, NvfVal, NvfTensorView>(
    std::function<NvfTensorView*(NvfVal*, NvfTensorView*)> func, NvfVal* arg1, NvfVal* arg2) {
  return func(arg1, arg2->as<NvfTensorView>());
}

template<class OutType, class Arg1Type, class Arg2Type>
struct BinaryOpRecord : RecordFunctor {
  BinaryOpRecord(std::vector<size_t> _args,
                std::vector<size_t> _outputs,
                std::function<OutType*(Arg1Type*, Arg2Type*)> fusion_op) :
    RecordFunctor(std::move(_args), std::move(_outputs)),
    fusion_op_(fusion_op) {}

  void operator()(FusionDefinition& fd) final {
    auto arg1 = fd.fusion_state.at(args.at(0));
    auto arg2 = fd.fusion_state.at(args.at(1));
    auto output = binary_op_func<OutType, Arg1Type, Arg2Type>(
                     fusion_op_, arg1, arg2);
    fd.fusion_state.at(outputs.at(0)) = output;
  }

 private:
  std::function<OutType*(Arg1Type*, Arg2Type*)> fusion_op_;
};

// With C++17, this template specialization could be replaced by
// "if constexpr" statement in the operator() 
// function.
template<class O, class A1, class A2>
O* binary_with_alpha_op_func(std::function<O*(A1*, A2*, NvfVal*)> func, NvfVal* arg1, NvfVal* arg2, NvfVal* arg3) {
  return func(arg1, arg2, arg3);
}
template<>
NvfTensorView* binary_with_alpha_op_func<NvfTensorView, NvfTensorView, NvfTensorView>(std::function<NvfTensorView*(NvfTensorView*, NvfTensorView*, NvfVal*)> func, NvfVal* arg1, NvfVal* arg2, NvfVal* arg3) {
  return func(arg1->as<NvfTensorView>(), arg2->as<NvfTensorView>(), arg3);
}
template<>
NvfTensorView* binary_with_alpha_op_func<NvfTensorView, NvfTensorView, NvfVal>(std::function<NvfTensorView*(NvfTensorView*, NvfVal*, NvfVal*)> func, NvfVal* arg1, NvfVal* arg2, NvfVal* arg3) {
  return func(arg1->as<NvfTensorView>(), arg2, arg3);
}
template<>
NvfTensorView* binary_with_alpha_op_func<NvfTensorView, NvfVal, NvfTensorView>(std::function<NvfTensorView*(NvfVal*, NvfTensorView*, NvfVal*)> func, NvfVal* arg1, NvfVal* arg2, NvfVal* arg3) {
  return func(arg1, arg2->as<NvfTensorView>(), arg3);
}

template<class OutType, class Arg1Type, class Arg2Type>
struct BinaryWithAlphaOpRecord : RecordFunctor {
  BinaryWithAlphaOpRecord(std::vector<size_t> _args,
                std::vector<size_t> _outputs,
                std::function<OutType*(Arg1Type*, Arg2Type*, NvfVal*)> fusion_op) :
    RecordFunctor(std::move(_args), std::move(_outputs)),
    fusion_op_(fusion_op) {}

  void operator()(FusionDefinition& fd) final {
    auto arg1 = fd.fusion_state.at(args.at(0));
    auto arg2 = fd.fusion_state.at(args.at(1));
    auto arg3 = fd.fusion_state.at(args.at(2));
    auto output = binary_with_alpha_op_func<OutType, Arg1Type, Arg2Type>(
                     fusion_op_, arg1, arg2, arg3);
    fd.fusion_state.at(outputs.at(0)) = output;
  }

 private:
  std::function<OutType*(Arg1Type*, Arg2Type*)> fusion_op_;
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
    
    fd.fusion_state.at(outputs.at(0)) = tv;
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
    auto input = fd.fusion_state.at(args.at(0));

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
    auto arg = fd.fusion_state.at(args.at(0));
    auto output = unary_op_func<ArgType>(fusion_op_, arg);
    fd.fusion_state.at(outputs.at(0)) = output;
  }

 private:
  std::function<ArgType*(ArgType*)> fusion_op_;
};

} // nvfuser namespace

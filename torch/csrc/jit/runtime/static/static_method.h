#pragma once

#include <torch/csrc/api/include/torch/imethod.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch::jit {

class StaticMethod : public torch::IMethod {
 public:
  StaticMethod(
      std::shared_ptr<StaticModule> static_module,
      std::string method_name)
      : static_module_(std::move(static_module)),
        method_name_(std::move(method_name)) {
    TORCH_CHECK(static_module_);
  }

  c10::IValue operator()(
      std::vector<IValue> args,
      const IValueMap& kwargs = IValueMap()) const override {
    return (*static_module_)(std::move(args), kwargs);
  }

  const std::string& name() const override {
    return method_name_;
  }

 protected:
  void setArgumentNames(
      std::vector<std::string>& argument_names_out) const override {
    const auto& schema = static_module_->schema();
    CAFFE_ENFORCE(schema.has_value());
    const auto& arguments = schema->arguments();
    argument_names_out.clear();
    argument_names_out.reserve(arguments.size());
    std::transform(
        arguments.begin(),
        arguments.end(),
        std::back_inserter(argument_names_out),
        [](const c10::Argument& arg) -> std::string { return arg.name(); });
  }

 private:
  std::shared_ptr<StaticModule> static_module_;
  std::string method_name_;
};

} // namespace torch::jit

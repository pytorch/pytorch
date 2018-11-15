#include <torch/nn/module.h>

#include <torch/ordered_dict.h>

#include <torch/csrc/autograd/generated/VariableType.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <typeinfo>

namespace torch {
namespace nn {
namespace {
/// Joins names hierarchically: "name_prefix.name" if `name_prefix` is
/// non-empty, else just "name".
std::string join_name(const std::string& name_prefix, const std::string& name) {
  size_t total_size = name.size();
  if (!name_prefix.empty()) {
    total_size += name_prefix.size() + 1;
  }
  std::string full_name;
  full_name.reserve(total_size);
  if (!name_prefix.empty()) {
    full_name += name_prefix;
    full_name.push_back('.');
  }
  full_name += name;
  return full_name;
}

void extend(
    std::vector<Tensor>& vector,
    const OrderedDict<std::string, Tensor>& dict) {
  vector.reserve(vector.size() + dict.size());
  for (const auto& item : dict) {
    vector.push_back(item.value());
  }
}
} // namespace

Module::Module()
    : parameters_("Parameter"), buffers_("Buffer"), children_("Submodule") {}

Module::Module(std::string name) : Module() {
  name_ = std::move(name);
}

const std::string& Module::name() const noexcept {
  // If the name optional is empty at this point, we grab the name of the
  // dynamic type via RTTI. Note that we cannot do this in the constructor,
  // because in the constructor of a base class `this` always refers to the base
  // type. Inheritance effectively does not work in constructors. Also this note
  // from http://en.cppreference.com/w/cpp/language/typeid:
  // If typeid is used on an object under construction or destruction (in a
  // destructor or in a constructor, including constructor's initializer list
  // or default member initializers), then the std::type_info object referred
  // to by this typeid represents the class that is being constructed or
  // destroyed even if it is not the most-derived class.
  if (!name_.has_value()) {
    name_ = c10::demangle(typeid(*this).name());
#if defined(_WIN32)
    // Windows adds "struct" or "class" as a prefix.
    if (name_->find("struct ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 7);
    } else if (name_->find("class ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 6);
    }
#endif // defined(_WIN32)
  }
  return *name_;
}

std::shared_ptr<Module> Module::clone(optional<Device> device) const {
  AT_ERROR(
      "clone() has not been implemented for ",
      name(),
      ". Subclass torch::nn::Cloneable<",
      name(),
      "> instead of torch::nn::Module to inherit the ability to clone.");
}

void Module::apply(ModuleApplyFunction function) {
  function(*this);
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

void Module::apply(ConstModuleApplyFunction function) const {
  function(*this);
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

void Module::apply(NamedModuleApplyFunction function, std::string name_prefix) {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](
          const std::string& name, const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      std::move(name_prefix));
}

void Module::apply(
    ConstNamedModuleApplyFunction function,
    std::string name_prefix) const {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](
          const std::string& name, const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      std::move(name_prefix));
}

void Module::apply(ModulePointerApplyFunction function) const {
  function(shared_from_this_checked());
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(module);
      });
}

void Module::apply(
    NamedModulePointerApplyFunction function,
    std::string name_prefix) const {
  function(
      /*name=*/name_prefix, shared_from_this_checked());
  apply_to_submodules(function, std::move(name_prefix));
}

std::vector<Tensor> Module::parameters(bool recurse) const {
  if (!recurse) {
    return parameters_.values();
  }
  std::vector<Tensor> result;
  apply(
      [&result](const Module& module) { extend(result, module.parameters_); });
  return result;
}

OrderedDict<std::string, Tensor> Module::named_parameters(bool recurse) const {
  if (!recurse) {
    return parameters_;
  }
  OrderedDict<std::string, Tensor> result;
  apply([&result](const std::string& name, const Module& module) {
    for (const auto& parameter : module.parameters_) {
      result.insert(join_name(name, parameter.key()), parameter.value());
    }
  });
  return result;
}

std::vector<Tensor> Module::buffers(bool recurse) const {
  if (!recurse) {
    return buffers_.values();
  }
  std::vector<Tensor> result;
  apply([&result](const Module& module) { extend(result, module.buffers_); });
  return result;
}
OrderedDict<std::string, Tensor> Module::named_buffers(bool recurse) const {
  if (!recurse) {
    return buffers_;
  }
  OrderedDict<std::string, Tensor> result;
  apply([&result](const std::string& name, const Module& module) {
    for (const auto& buffer : module.buffers_) {
      result.insert(join_name(name, buffer.key()), buffer.value());
    }
  });
  return result;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool include_self) const {
  std::vector<std::shared_ptr<Module>> result;
  if (include_self) {
    apply([&result](const std::shared_ptr<Module>& module) {
      result.push_back(module);
    });
  } else {
    apply_to_submodules(
        [&result](const std::string&, const std::shared_ptr<Module>& module) {
          result.push_back(module);
        });
  }
  return result;
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_modules(
    std::string name_prefix,
    bool include_self) const {
  OrderedDict<std::string, std::shared_ptr<Module>> result;
  if (include_self) {
    apply(
        [&result](
            const std::string& key, const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        std::move(name_prefix));
  } else {
    apply_to_submodules(
        [&result](
            const std::string& key, const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        std::move(name_prefix));
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::children() const {
  return children_.values();
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_children()
    const {
  return children_;
}

void Module::train() {
  for (auto& child : children_) {
    child.value()->train();
  }
  is_training_ = true;
}

void Module::eval() {
  for (auto& child : children_) {
    child.value()->eval();
  }
  is_training_ = false;
}

void Module::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

void Module::to(torch::Dtype dtype, bool non_blocking) {
  to_impl(dtype, non_blocking);
}

void Module::to(torch::Device device, bool non_blocking) {
  to_impl(device, non_blocking);
}

bool Module::is_training() const noexcept {
  return is_training_;
}

void Module::zero_grad() {
  for (auto& child : children_) {
    child.value()->zero_grad();
  }
  for (auto& parameter : parameters_) {
    auto& grad = parameter->grad();
    if (grad.defined()) {
      grad = grad.detach();
      grad.zero_();
    }
  }
}

void Module::save(serialize::OutputArchive& archive) const {
  for (const auto& parameter : parameters_) {
    archive.write(parameter.key(), parameter.value());
  }
  for (const auto& buffer : buffers_) {
    archive.write(buffer.key(), buffer.value(), /*is_buffer=*/true);
  }
  for (const auto& child : children_) {
    serialize::OutputArchive child_archive;
    child.value()->save(child_archive);
    archive.write(child.key(), child_archive);
  }
}

void Module::load(serialize::InputArchive& archive) {
  for (auto& parameter : parameters_) {
    archive.read(parameter.key(), parameter.value());
  }
  for (auto& buffer : buffers_) {
    archive.read(buffer.key(), buffer.value(), /*is_buffer=*/true);
  }
  for (const auto& child : children_) {
    // Modules that have no state at all (parameters or buffers) are currently
    // not stored in Protobuf at all, so we can just skip them.
    if (!child.value()->parameters_.is_empty() ||
        !child.value()->buffers_.is_empty()) {
      serialize::InputArchive child_archive;
      archive.read(child.key(), child_archive);
      child.value()->load(child_archive);
    }
  }
}

Tensor& Module::register_parameter(
    std::string name,
    Tensor tensor,
    bool requires_grad) {
  AT_CHECK(!name.empty(), "Parameter name must not be empty");
  AT_CHECK(
      name.find('.') == std::string::npos,
      "Parameter name must not contain a dot (got '",
      name,
      "')");
  tensor.set_requires_grad(requires_grad);
  return parameters_.insert(std::move(name), std::move(tensor));
}

Tensor& Module::register_buffer(std::string name, Tensor tensor) {
  AT_CHECK(!name.empty(), "Buffer name must not be empty");
  AT_CHECK(
      name.find('.') == std::string::npos,
      "Buffer name must not contain a dot (got '",
      name,
      "')");
  return buffers_.insert(std::move(name), std::move(tensor));
}

void Module::clone_(Module& other, optional<Device> device) {}

void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    std::string name_name_prefix) const {
  for (const auto& child : children_) {
    auto qualified_name = join_name(name_name_prefix, child.key());
    function(qualified_name, child.value());
    child.value()->apply_to_submodules(function, std::move(qualified_name));
  }
}

std::shared_ptr<Module> Module::shared_from_this_checked() const {
  std::shared_ptr<const Module> ptr;
  try {
    ptr = shared_from_this();
  } catch (const std::bad_weak_ptr& e) {
    AT_ERROR(
        "It looks like you attempted to retrieve your top-level module "
        "as a shared_ptr, but it is not stored in a shared_ptr. "
        "Use std::make_shared<",
        name(),
        "> instead of creating your module on "
        "the stack, or alternatively do not try to access your top-level "
        "module at all by passing /*include_self=*/false "
        "to modules() or named_modules()");
  }
  return std::const_pointer_cast<Module>(ptr);
}
} // namespace nn
} // namespace torch

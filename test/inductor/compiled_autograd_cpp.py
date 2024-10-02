test_autograd_cpp_node = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
    static constexpr bool is_traceable = true;

    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x) {
        return x;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output) {
        return grad_output;
    }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
    return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

test_autograd_cpp_node_id = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

struct CustomOpAutogradFunction2 : public torch::autograd::Function<CustomOpAutogradFunction2> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

torch::Tensor custom_op_backed_by_autograd_fn2(torch::Tensor x) {
  return CustomOpAutogradFunction2::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_id, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
"""

test_autograd_cpp_node_saved = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y,
      const torch::Tensor& fixed) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["fixed_tensor"] = fixed;
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;
    c10::List<std::string> list({"string"});
    ctx->saved_data["list"] = std::move(list);
    c10::Dict<std::string, double> dict;
    dict.insert("string", 1.0);
    ctx->saved_data["dict"] = std::move(dict);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    torch::Tensor fixed = ctx->saved_data["fixed_tensor"].toTensor();
    assert(ctx->saved_data["bool"].isBool());
    c10::SymInt i = ctx->saved_data["int"].toSymInt();
    c10::List<c10::IValue> list = ctx->saved_data["list"].toList();
    assert(list.size() == 1);
    assert(list.get(0).toStringRef() == "string");
    c10::Dict<c10::IValue, c10::IValue> dict = ctx->saved_data["dict"].toGenericDict();
    assert(dict.size() == 1);
    assert(dict.at("string") == 1.0);

    torch::autograd::variable_list grad_inputs(3);
    grad_inputs[0] = x + y + torch::sum(fixed) + i;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& fixed) {
  return CustomOpAutogradFunction::apply(x, y, fixed);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

test_autograd_cpp_node_saved_dynamic = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    ctx->save_for_backward({x});
    ctx->saved_data["dynamic"] = x.view(-1);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    torch::Tensor z = ctx->saved_data["dynamic"].toTensor();

    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = x + torch::sum(z);
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""


test_autograd_cpp_node_saved_int = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      int64_t y) {
    ctx->save_for_backward({x});
    ctx->saved_data["int"] = y;
    ctx->saved_data["symint"] = c10::SymInt(y);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymInt y = ctx->saved_data["int"].toSymInt();
    c10::SymInt ys = ctx->saved_data["symint"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + ys;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, int64_t y) {
  return CustomOpAutogradFunction::apply(x, y);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_int, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

test_autograd_cpp_node_saved_float = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      double z) {
    ctx->save_for_backward({x});
    ctx->saved_data["float"] = z;
    ctx->saved_data["symfloat"] = c10::SymFloat(z);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymFloat z = ctx->saved_data["float"].toSymFloat();
    c10::SymFloat zs = ctx->saved_data["symfloat"].toSymFloat();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + z + zs;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, double z) {
  return CustomOpAutogradFunction::apply(x, z);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_float, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

test_autograd_cpp_node_data_dependent = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;
  static int iteration;

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;

    switch (iteration) {
        case 0: {
            break;
        }
        case 1: {
            // recompile
            ctx->saved_data["forces_recompile"] = iteration;
            break;
        }
        case 2: {
            // recompile
            ctx->set_materialize_grads(false);
            break;
        }
        case 3: {
            // reuse
            break;
        }
        default: {
            throw std::runtime_error("unexpected iteration");
        }
    }
    iteration++;
    return {x, y};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    c10::SymInt i = ctx->saved_data["int"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + i;
    return grad_inputs;
  }
};

int CustomOpAutogradFunction::iteration = 0;

torch::autograd::variable_list custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y) {
  return CustomOpAutogradFunction::apply(x, y);
}

void reset() {
    CustomOpAutogradFunction::iteration = 0;
}

TORCH_LIBRARY(test_autograd_cpp_node_data_dependent, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("reset", reset);
}
"""

test_autograd_cpp_node_non_traceable = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = false;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_non_traceable_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""



test_cudagraphs_cpu_scalar_used_in_cpp_custom_op = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    const auto& cpu_tensor = torch::tensor(1);
    ctx->save_for_backward({x, cpu_tensor});
    ctx->saved_data["cpu_scalar"] = 1;
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor cpu_tensor = saved_variables[1];
    int cpu_scalar = ctx->saved_data["cpu_scalar"].toInt();
    auto expand = grad_output[0] * torch::ones_like(x);
    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = expand * cpu_tensor * cpu_scalar;  // autograd engine asserts that tensors are on same device
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_cudagraphs_cpu_scalar_used_in_cpp_custom_op, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

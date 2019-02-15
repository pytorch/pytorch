#include <torch/csrc/jit/symbolic_script.h>

namespace torch {
namespace jit {
namespace {
std::mutex lock;
const std::vector<std::string> functions = {
    R"(

        def _dim_arange(like,
                        dim: int):
            def backward(grad_output):
                return None, None

            return torch._dim_arange(like, dim), backward

        def contiguous(self):
            def backward(grad_output):
                return None

            return self.contiguous(), backward

        def erf(self):
            def backward(grad_output):
                # Precomputed constant C = 2.0 / math.sqrt(math.pi)
                C = 1.1283791670955126
                grad_self =  C * torch.exp(- self.pow(2)) * grad_output
                return grad_self

            return torch.erf(self), backward

        def expand(self,
                   size: List[int],
                   implicit: bool=False):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch._grad_sum_to_size(grad_output, self_size)
                return grad_self, None, None

            return torch.expand(self, size, implicit=implicit), backward

        def expand_as(self, other):
            self_size = self.size()
            def backward(grad_output):
                grad_self = grad_output._grad_sum_to_size(self_size)
                return grad_self, None

            return torch.expand_as(self, other), backward

        def full_like(self,
                      fill_value: float):
            def backward(grad_output):
                return None, None

            return torch.full_like(self, fill_value), backward

        def kthvalue(self,
                     k: int,
                     dim: int,
                     keepdim: bool):
            result0, result1 = torch.kthvalue(self, k, dim, keepdim)
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.index_select_backward(grad_output, dim, result1, self_size, keepdim)
                return grad_self, None, None, None

            return result0, result1, backward

        def logsumexp(self,
                      dim: List[int],
                      keepdim: bool):
            result = torch.logsumexp(self, dim, keepdim)
            self_dim = self.dim()
            def backward(grad_output):
                grad_self = torch.logsumexp_backward(grad_output, self, result, dim, keepdim)
                return grad_self, None, None

            return result, backward

        def mean_0(self):
            self_size = self.size()
            self_numel = self.numel()
            def backward(grad_output):
                grad_self = grad_output.expand(self_size) / self_numel
                return grad_self

            return torch.mean(self), backward

        def mean_1(self,
                   dim: List[int],
                   keepdim: bool):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.sum_backward(grad_output, self_size, dim, keepdim) / torch._safe_size(self_size, dim)
                return grad_self, None, None

            return torch.mean(self, dim, keepdim), backward

        def mul(self, other):
            self_size = self.size()
            other_size = other.size()
            def backward(grad_output):
                grad_self = (grad_output * other)._grad_sum_to_size(self_size)
                grad_other = (grad_output * self)._grad_sum_to_size(other_size)
                return grad_self, grad_other

            return self * other, backward

        def nonzero(self):
            def backward(grad_output):
                return None

            return torch.nonzero(self), backward

        def ones_like(self):
            def backward(grad_output):
                return None

            return torch.ones_like(self), backward

        def permute(self,
                    dims: List[int]):
            def backward(grad_output):
                grad_self = torch.permute_backwards(grad_output, dims)
                return grad_self, None

            return torch.permute(self, dims), backward

        def pow_0(self,
                  exponent: float):
            def backward(grad_output):
                grad_self = torch.where(torch.tensor(exponent == 0.0), torch.zeros_like(self), grad_output * exponent * torch.pow(self, exponent - 1))
                return grad_self, None

            return torch.pow(self, exponent), backward

        def pow_1(self, exponent):
            self_size = self.size()
            exponent_size = exponent.size()
            def backward(grad_output):
                grad_self = torch.where(exponent == 0.0, torch.zeros_like(self), grad_output * exponent * torch.pow(self, exponent - 1))._grad_sum_to_size(self_size)
                grad_exponent = (grad_output * torch.pow(self, exponent) * torch.log(self))._grad_sum_to_size(exponent_size)
                return grad_self, grad_exponent

            return torch.pow(self, exponent), backward

        def pow_2(self: float,
                  exponent):
            def backward(grad_output):
                grad_exponent = grad_output * torch.pow(self, exponent) * torch.log(torch.tensor(self))
                return None, grad_exponent

            return torch.pow(self, exponent), backward

        def rsub_0(self, other,
                   alpha: float = 1.0):
            self_size = self.size()
            other_size = other.size()
            def backward(grad_output):
                grad_self = (- grad_output * alpha)._grad_sum_to_size(self_size)
                grad_other = (grad_output)._grad_sum_to_size(other_size)
                return grad_self, grad_other, None

            return torch.rsub(self, other, alpha), backward

        def rsub_1(self,
                   other: float,
                   alpha: float = 1.0):
            self_size = self.size()
            def backward(grad_output):
                grad_self = (- grad_output * alpha)._grad_sum_to_size(self_size)
                return grad_self, None, None

            return torch.rsub(self, other, alpha), backward

        def select(self,
                   dim: int,
                   index: int):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.select_backward(grad_output, self_size, dim, index)
                return grad_self, None, None

            return torch.select(self, dim, index), backward

        def sqrt(self):
            result = torch.sqrt(self)
            def backward(grad_output):
                grad_self = grad_output / (2 * result)
                return grad_self

            return result, backward

        def squeeze_0(self):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.unsqueeze_to(grad_output, self_size)
                return grad_self

            return torch.squeeze(self), backward

        def squeeze_1(self,
                      dim: int):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.unsqueeze_to(grad_output, dim, self_size)
                return grad_self, None

            return torch.squeeze(self, dim), backward

        def t(self):
            def backward(grad_output):
                grad_self = torch.t(grad_output)
                return grad_self

            return torch.t(self), backward

        def to_0(self,
                 device: Optional[Device],
                 dtype: Optional[int],
                 non_blocking: bool=False,
                 copy: bool=False):
            self_device = self.device
            self_dtype = self.dtype
            if device is not None:
                result = self.to(device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            else:
                result = self.to(dtype, non_blocking=non_blocking, copy=copy)
            def backward(grad_output):
                grad_self = grad_output.to(self_device, dtype=self_dtype, non_blocking=non_blocking, copy=copy)
                return grad_self, None, None, None, None

            return result, backward


        def to_1(self,
                 dtype: int,
                 non_blocking: bool=False,
                 copy: bool=False):
            self_dtype = self.dtype
            def backward(grad_output):
                grad_self = grad_output.to(self_dtype, non_blocking, copy)
                return grad_self, None, None, None

            return self.to(dtype=dtype, non_blocking=non_blocking, copy=copy), backward

        def to_2(self,
                 other,
                 non_blocking: bool=False,
                 copy: bool=False):
            def backward(grad_output):
                grad_self = grad_output.to(self, non_blocking, copy)
                return grad_self, None, None, None

            return self.to(other, non_blocking=non_blocking, copy=copy), backward

        def topk(self,
                 k,
                 dim: int = -1,
                 largest: bool = True,
                 sorted: bool = True):
            result0, result1 = torch.topk(self, k, dim, largest, sorted)
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch.index_select_backward(grad_output, dim, result1, self_size, True)
                return grad_self, None, None, None, None

            return result0, result1, backward

        def transpose(self,
                      dim0: int,
                      dim1: int):
            def backward(grad_output):
                grad_self = torch.transpose(grad_output, dim0, dim1)
                return grad_self, None, None

            return torch.transpose(self, dim0, dim1), backward

        def var_0(self,
                  unbiased: bool=True):
            def backward(grad_output):
                grad_self = torch.var_backward(grad_output, self, unbiased)
                return grad_self, None

            return torch.var(self, unbiased), backward

        def var_1(self,
                  dim: List[int],
                  unbiased: bool,
                  keepdim: bool):
            def backward(grad_output):
                grad_self = torch.var_backward(grad_output, self, dim, unbiased, keepdim)
                return grad_self, None, None, None

            return torch.var(self, dim, unbiased, keepdim), backward

        def std_0(self,
                  unbiased: bool=True):
            std_out = torch.std(self, unbiased)
            def backward(grad_output):
                grad_self = torch.var_backward(grad_output / (std_out * 2), self, unbiased)
                return grad_self, None

            return std_out, backward

        def std_1(self,
                  dim: List[int],
                  unbiased: bool,
                  keepdim: bool):
            std_out = torch.std(self, dim, unbiased, keepdim)
            def backward(grad_output):
                grad_self = torch.var_backward(grad_output / (std_out * 2), self, dim, unbiased, keepdim)
                return grad_self, None, None, None

            return std_out, backward

        def view(self,
                 size: List[int]):
            self_size = self.size()
            def backward(grad_output):
                grad_self = grad_output.reshape(self_size)
                return grad_self, None

            return torch.view(self, size), backward

        def _adaptive_avg_pool2d(self,
                                output_size: List[int]):
            def backward(grad_output):
                grad_self = torch._adaptive_avg_pool2d_backward(grad_output, self)
                return grad_self, None

            return torch._adaptive_avg_pool2d(self, output_size), backward

        def embedding(weight,
                      indices,
                      padding_idx: int,
                      scale_grad_by_freq: bool,
                      sparse: bool):
            weight_size_0 = weight.size()[0]
            def backward(grad_output):
                grad_weight = torch.embedding_backward(grad_output, indices, weight_size_0, padding_idx, scale_grad_by_freq, sparse)
                return grad_weight, None, None, None, None

            return torch.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse), backward

      )"};
std::unordered_map<std::string, GradientPair> schema_to_graphs;

// This map is a workaround to cache compiled gradient_pairs. Ideally this graph
// should be compiled only once and saved in Operator structure.
// This should be done along with merging into native_functions.yaml.
std::unordered_map<const FunctionSchema*, GradientPair> cached_gradient_pairs;
} // anonymous namespace

std::pair<std::shared_ptr<Graph>, Value*> extractClosure(Value* closure) {
  AT_CHECK(
      closure->node()->kind() == prim::TupleConstruct,
      "closure must be a literal tuple construct");
  Value* fn = closure->node()->inputs().at(0);
  Value* context = closure->node()->inputs().at(1);

  AT_CHECK(
      fn->node()->kind() == prim::Function,
      "closure tuple must contain a prim::Function");
  return std::make_pair(fn->node()->g(attr::Subgraph), context);
}

Argument originalReturnType(const TupleTypePtr& tup) {
  AT_CHECK(tup->elements().size() > 1);
  if (tup->elements().size() == 2)
    return Argument("", tup->elements().at(0));
  std::vector<TypePtr> types = tup->elements().vec();
  types.pop_back();
  return Argument("", TupleType::create(std::move(types)));
}

// In torchscript AD formulas, we define {func_0, func_1, ...} as
// overloaded functions of `func`.
// Remove the suffix before adding the schema string to map
// schema_to_graphs.
std::string overloadedSchemaString(const FunctionSchema& schema) {
  const auto& schema_name = schema.name();
  auto pos = schema_name.find_last_of('_');
  auto schema_name_suffix = schema_name.substr(pos + 1);
  std::string schema_string = canonicalSchemaString(schema);
  if (!schema_name_suffix.empty()
      && schema_name_suffix.find_first_not_of("0123456789") == string::npos) {
    schema_string.replace(schema_string.find(schema_name),
                          schema_name.length(),
                          schema_name.substr(0, pos));
  }
  return schema_string;
}

void loadModule(const std::shared_ptr<script::Module>& module) {
  for (const auto& method_ : module->get_methods()) {
    const auto& method = method_.value();
    GradientPair pair;
    pair.forward = method->graph();

    // lookup the backward function
    Node* forward_tuple = pair.forward->outputs().at(0)->node();

    if (forward_tuple->kind() != prim::TupleConstruct) {
      throw script::ErrorReport(forward_tuple->getSourceLocation())
          << "gradient must return literal a tuple";
    }

    Value* context;
    std::tie(pair.backward, context) =
        extractClosure(forward_tuple->inputs().back());

    // do surgery on the forward function to remove the closure tuple and
    // replace it with the context variable:
    //  backward = (<lambda>, context_tuple)
    //  return original, backward
    //  -----
    //  return original, context_tuple
    std::vector<Value*> new_inputs = forward_tuple->inputs().vec();
    new_inputs.back() = context;
    Value* new_tuple =
        pair.forward->appendNode(pair.forward->createTuple(new_inputs))
            ->output();
    pair.forward->eraseOutput(0);
    pair.forward->registerOutput(new_tuple);
    forward_tuple->destroy();

    // derive schema from original function's schema:
    const FunctionSchema& loaded_schema = method->getSchema();
    FunctionSchema actual_schema(
        Symbol::aten(loaded_schema.name()),
        loaded_schema.arguments(),
        {originalReturnType(new_tuple->type()->expect<TupleType>())});

    // modify canonical string for function overloading
    // prefer not to modify the schema name
    auto schema_string = overloadedSchemaString(actual_schema);

    schema_to_graphs[schema_string] = std::move(pair);
  }
}

void loadFunctions() {
  for (const std::string& str : functions) {
    auto cu = std::make_shared<script::Module>();
    script::defineMethodsInModule(cu, str, script::nativeResolver, nullptr);
    loadModule(cu);
  }
}

c10::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (schema_to_graphs.size() == 0) {
    loadFunctions();
  }
  auto cache_it = cached_gradient_pairs.find(&schema);
  if (cache_it != cached_gradient_pairs.end()) {
    return cache_it->second;
  } else {
    auto schema_str = canonicalSchemaString(schema);
    // JIT doesn't support keyword only arguments.
    // Remove ' *,' in schema before looking up
    // TODO: #16921 properly support keyword only arguments in JIT.
    auto n = schema_str.find("*, ");
    if (n != std::string::npos) {
      schema_str = schema_str.erase(n, 3);
    }

    auto sym_script_it = schema_to_graphs.find(schema_str);
    if (sym_script_it != schema_to_graphs.end()) {
      cached_gradient_pairs.emplace_hint(
          cache_it, &schema, sym_script_it->second);
      return sym_script_it->second;
    }
  }
  return c10::nullopt;
}

bool hasGradientInfoForSchema(const FunctionSchema& schema) {
  return gradientInfoForSchema(schema).has_value();
}

} // namespace jit
} // namespace torch

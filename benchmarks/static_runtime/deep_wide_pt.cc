#include "deep_wide_pt.h"

#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/script.h>

namespace {
// No ReplaceNaN (this removes the constant in the model)
const std::string deep_wide_pt = R"JIT(
class DeepAndWide(Module):
  __parameters__ = ["_mu", "_sigma", "_fc_w", "_fc_b", ]
  __buffers__ = []
  _mu : Tensor
  _sigma : Tensor
  _fc_w : Tensor
  _fc_b : Tensor
  training : bool
  def forward(self: __torch__.DeepAndWide,
    ad_emb_packed: Tensor,
    user_emb: Tensor,
    wide: Tensor) -> Tuple[Tensor]:
    _0 = self._fc_b
    _1 = self._fc_w
    _2 = self._sigma
    wide_offset = torch.add(wide, self._mu, alpha=1)
    wide_normalized = torch.mul(wide_offset, _2)
    wide_preproc = torch.clamp(wide_normalized, 0., 10.)
    user_emb_t = torch.transpose(user_emb, 1, 2)
    dp_unflatten = torch.bmm(ad_emb_packed, user_emb_t)
    dp = torch.flatten(dp_unflatten, 1, -1)
    input = torch.cat([dp, wide_preproc], 1)
    fc1 = torch.addmm(_0, input, torch.t(_1), beta=1, alpha=1)
    return (torch.sigmoid(fc1),)
)JIT";

const std::string trivial_model_1 = R"JIT(
  def forward(self, a, b, c):
      s = torch.tensor([[3, 3], [3, 3]])
      return a + b * c + s
)JIT";

const std::string leaky_relu_model_const = R"JIT(
  def forward(self, input):
      x = torch.leaky_relu(input, 0.1)
      x = torch.leaky_relu(x, 0.1)
      x = torch.leaky_relu(x, 0.1)
      x = torch.leaky_relu(x, 0.1)
      return torch.leaky_relu(x, 0.1)
)JIT";

const std::string leaky_relu_model = R"JIT(
  def forward(self, input, neg_slope):
      x = torch.leaky_relu(input, neg_slope)
      x = torch.leaky_relu(x, neg_slope)
      x = torch.leaky_relu(x, neg_slope)
      x = torch.leaky_relu(x, neg_slope)
      return torch.leaky_relu(x, neg_slope)
)JIT";


void import_libs(
    std::shared_ptr<at::CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<torch::jit::Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  torch::jit::SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& /* unused */) -> std::shared_ptr<torch::jit::Source> {
        return src;
      },
      /*version=*/2);
  si.loadType(c10::QualifiedName(class_name));
}
} // namespace

torch::jit::Module getDeepAndWideSciptModel(int num_features) {
  auto cu = std::make_shared<at::CompilationUnit>();
  std::vector<at::IValue> constantTable;
  import_libs(
      cu,
      "__torch__.DeepAndWide",
      std::make_shared<torch::jit::Source>(deep_wide_pt),
      constantTable);
  c10::QualifiedName base("__torch__");
  auto clstype = cu->get_class(c10::QualifiedName(base, "DeepAndWide"));

  torch::jit::Module mod(cu, clstype);

  mod.register_parameter("_mu", torch::randn({1, num_features}), false);
  mod.register_parameter("_sigma", torch::randn({1, num_features}), false);
  mod.register_parameter("_fc_w", torch::randn({1, num_features + 1}), false);
  mod.register_parameter("_fc_b", torch::randn({1}), false);

  // mod.dump(true, true, true);
  return mod;
}

torch::jit::Module getTrivialScriptModel() {
  torch::jit::Module module("m");
  module.define(trivial_model_1);
  return module;
}

torch::jit::Module getLeakyReLUScriptModel() {
  torch::jit::Module module("leaky_relu");
  module.define(leaky_relu_model);
  return module;
}

torch::jit::Module getLeakyReLUConstScriptModel() {
  torch::jit::Module module("leaky_relu_const");
  module.define(leaky_relu_model_const);
  return module;
}

const std::string long_model = R"JIT(
  def forward(self, a, b, c):
      d = torch.relu(a * b)
      e = torch.relu(a * c)
      f = torch.relu(e * d)
      g = torch.relu(f * f)
      h = torch.relu(g * c)
      return h
)JIT";

torch::jit::Module getLongScriptModel() {
  torch::jit::Module module("m");
  module.define(long_model);
  return module;
}

#pragma once

#include <torch/torch.h>

const auto list_construct_script = R"JIT(
  def forward(self, a, b):
    return [a, b]
)JIT";

const auto list_unpack_script = R"JIT(
  def forward(self, a, b):
    c = [a, b]
    x, y = c
    z = x + y
    return z
)JIT";

const auto tuple_construct_script = R"JIT(
  def forward(self, a, b):
    return (a, b)
)JIT";

const auto add_script = R"JIT(
  def forward(self, a, b):
      return a + b
)JIT";

const auto reshape_script_1 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.reshape(shape)
      return b + b
)JIT";

const auto reshape_script_2 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.transpose(0, 1)
      return b.reshape(shape)
)JIT";

const auto flatten_script_1 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = torch.flatten(a, start_dim, end_dim)
      return b + b
)JIT";

const auto flatten_script_2 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = a.transpose(0, 1)
      return torch.flatten(b, start_dim, end_dim)
)JIT";

#pragma once
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

// TODO: the internals here might change in the future, this
// is just a placeholder
static std::unordered_map<std::string, std::string> upgraders_graph(
    {{"div_Tensor_0_3", R"IR(graph(%self.1 : Tensor,
                                  %other.1 : Tensor):
                            %32 : str = prim::Constant[value="trunc"]()
                            %6 : bool = prim::Constant[value=1]()
                            %4 : bool = aten::is_floating_point(%self.1)
                            %11 : bool = prim::If(%4)
                                block0():
                                    -> (%6)
                                block1():
                                    %9 : bool = aten::is_floating_point(%other.1)
                                    -> (%9)
                            %35 : Tensor = prim::If(%11)
                                block0():
                                    %36 : Tensor = aten::div(%self.1, %other.1)
                                    -> (%36)
                                block1():
                                    %37 : Tensor = aten::div(%self.1, %other.1, %32)
                                    -> (%37)
                            return (%35))IR"},
     {"div_Scalar_0_3", R"IR(graph(%self.1 : Tensor,
                                %other.1 : Scalar):
                            %41 : str = prim::Constant[value=\"trunc\"]()
                            %6 : bool = prim::Constant[value=1]()
                            %4 : bool = aten::is_floating_point(%self.1)
                            %9 : bool = prim::If(%4)
                                block0():
                                    -> (%6)
                                block1():
                                    %8 : bool = prim::isinstance[types=[float]](%other.1)
                                    -> (%8)
                            %44 : Tensor = prim::If(%9) # torch/jit/operator_upgraders.py:21:4
                                block0():
                                    %45 : Tensor = aten::div(%self.1, %other.1) # torch/jit/operator_upgraders.py:22:15
                                    -> (%45)
                                block1():
                                    %other.9 : Union[complex, int] = prim::unchecked_cast(%other.1)
                                    %46 : Tensor = aten::div(%self.1, %other.9, %41) # torch/jit/operator_upgraders.py:23:11
                                    -> (%46)
                            return (%44))IR"},
     {"div_out_0_3", R"IR(graph(%self.1 : Tensor,
                            %other.1 : Tensor,
                            %out.1 : Tensor):
                        %41 : str = prim::Constant[value="trunc"]() # torch/jit/operator_upgraders.py:33:44
                        %7 : bool = prim::Constant[value=1]() # torch/jit/operator_upgraders.py:31:8
                        %5 : bool = aten::is_floating_point(%self.1) # torch/jit/operator_upgraders.py:31:8
                        %12 : bool = prim::If(%5) # torch/jit/operator_upgraders.py:31:8
                            block0():
                                -> (%7)
                            block1():
                            %10 : bool = aten::is_floating_point(%other.1) # torch/jit/operator_upgraders.py:31:36
                                -> (%10)
                        %18 : bool = prim::If(%12) # torch/jit/operator_upgraders.py:31:8
                            block0():
                                -> (%7)
                            block1():
                                %16 : bool = aten::is_floating_point(%out.1) # torch/jit/operator_upgraders.py:31:65
                                -> (%16)
                        %44 : Tensor = prim::If(%18) # torch/jit/operator_upgraders.py:31:4
                            block0():
                                %45 : Tensor = aten::div(%self.1, %other.1, %out.1) # torch/jit/operator_upgraders.py:32:15
                                -> (%45)
                            block1():
                                %46 : Tensor = aten::div(%self.1, %other.1, %41, %out.1) # torch/jit/operator_upgraders.py:33:11
                                -> (%46)
                        return (%44))IR"}});

} // namespace jit
} // namespace torch

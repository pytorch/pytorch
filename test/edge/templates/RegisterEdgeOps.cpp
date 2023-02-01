#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
/**
 * $generated_comment
 */
namespace torch::jit {
    using at::ScalarType;
    /*
    RegisterOperators reg({
            Operator("edge::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
                    [](Stack &stack) {
                        c10::OperatorName name = c10::OperatorName("aten::add", "Tensor");
                        std::shared_ptr<Operator> op = findOperatorFor(name);
                        if (op) {
                            op->getOperation()(stack);
                        }
                    },
                    aliasAnalysisFromSchema(),
                     {{"a", {ScalarType::Long, ScalarType::Int}}})
    });
     */

    RegisterOperators reg({
        $operators
    });
} // namespace torch::jit

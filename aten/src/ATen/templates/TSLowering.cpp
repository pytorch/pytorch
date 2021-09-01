// ${generated_comment}
${ts_lowering_sysinc}
${ts_lowering_inc}
namespace torch_lazy_tensors {
namespace compiler {

// I copied these for now but they really should be moved.
TSOpVector LowerBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    auto builtin =
        std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
    auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
    auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
    auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
    LTC_CHECK(sv);
    if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
        const auto tuple_call_result = sv->asTuple({}, *function);
        TSOpVector tuple_result;
        for (const auto& tuple_component : tuple_call_result) {
        auto tuple_component_sv =
            dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
        tuple_result.push_back(tuple_component_sv->getValue());
        }
        return tuple_result;
    }
    return {sv->getValue()};
}

TSOpVector LowerBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    const ir::Node* node,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
return LowerBuiltin(function, node->op().op, arguments, kwarguments);
}


${lowering_definitions}

TSOpVector LowerToTSCodegen(std::shared_ptr<torch::jit::GraphFunction> function,
                            ts_backend::TSLoweringContext* loctx,
                            const ir::Node* node) {
    switch (node->op().op){
${lowering_dispatches}
default:
    return {};
    }
}

} // namespace compiler
} // namespace torch_lazy_tensors


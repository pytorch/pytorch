// ${generated_comment}
${ts_lowering_sysinc}
${ts_lowering_inc}
namespace torch_lazy_tensors {
namespace compiler {
// namespace ${backend_namespace} {

TSOpVector LowerBuiltin(
      const ir::Node* node,
      const std::vector<torch::jit::NamedValue>& arguments) {
    // TODO
    return {};  
}

ts_backend::TSLoweringContext* loctx() {
    // TODO - need to integrate these generated methods below with the 
    // real TSLowering class that has the context.
    return nullptr;
}

${lowering_definitions}

TSOpVector LowerToTS(const ir::Node* node) {
    switch (node->op().op){
${lowering_dispatches}
    }
}

// } // namespace ${backend_namespace}
} // namespace compiler
} // namespace torch_lazy_tensors


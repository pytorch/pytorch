from cpp_api_parity import CppArgDeclaration

module_metadata_map = {
    'Linear': dict(
        cpp_forward_arg_declarations=[CppArgDeclaration(arg_type='const torch::Tensor&', arg_name='input')]
    )
}

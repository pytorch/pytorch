import torch
model = torch.load("bt.pt")

ii = model._generate_bundled_inputs_for_forward()
print([x.dtype for x in list(*ii)])
print([x.size() for x in list(*ii)])

model.eval()
model = torch.jit.freeze(model)
model._c.dump(attrs=False, params=False)

g = model.graph

# Before:
# graph(%self : __torch__.___torch_mangle_149.ModelProd, %bytes.1 : Tensor, %lens : Tensor):
#   ...
# After:
# graph(%bytes.1 : Tensor, %lens : Tensor)
#   ...
torch._C._te.remove_unused_self_argument(g)

# Inject shape/dtype/device info into the graph
g = torch._C._jit_trace_graph(g, tuple(*ii))
torch._C._te.annotate_input_shapes(g, list(*ii))

# Run a couple of cleanup passes
torch._C._jit_pass_remove_mutation(g)
torch._C._jit_pass_dce(g)

# Perform some hacky graph rewriting to make sure the outputs are just tensors and not lists/lists of strings/tuples
#
# First, get rid of a second element in the tuple which is always the same list of strings
#
# Before:
#   ...
#   %6 : str[] = prim::Constant[value=["suitable", "adulthealth", "wfh", "weapon", "unsubstantiatedclaim", "adfarm", "language", "lowqualityecommerce", "adultcontent", "tobacco", "financial", "restrictedfinancial"]]()
#   ...
#   %57 : Float(1, 32, strides=[32, 1], requires_grad=0, device=cpu) = prepacked::linear_clamp_run(%56, %28)
#   %ret.1 : Float(1, 12, strides=[12, 1], requires_grad=0, device=cpu) = aten::softmax(%59, %9, %30)
#   %tensors.1 : Tensor[] = prim::ListConstruct(%ret.1, %57)
#   %64 : (Tensor[], str[]) = prim::TupleConstruct(%tensors.1, %6)
#   return (%64)
# After:
#   ...
#   %57 : Float(1, 32, strides=[32, 1], requires_grad=0, device=cpu) = prepacked::linear_clamp_run(%56, %28)
#   %ret.1 : Float(1, 12, strides=[12, 1], requires_grad=0, device=cpu) = aten::softmax(%59, %9, %30)
#   %tensors.1 : Tensor[] = prim::ListConstruct(%ret.1, %57)
#   return (%tensors.1)
torch._C._jit_pass_lower_all_tuples(g)
torch._C._te.remove_graph_output(g, 1)
torch._C._jit_pass_dce(g)

# Second, replace the list of two elements with a tuple of two elements and then replace returning a tuple with simply returning two tensors.
#
# Before:
#   ...
#   %57 : Float(1, 32, strides=[32, 1], requires_grad=0, device=cpu) = prepacked::linear_clamp_run(%56, %28)
#   %ret.1 : Float(1, 12, strides=[12, 1], requires_grad=0, device=cpu) = aten::softmax(%59, %9, %30)
#   %tensors.1 : Tensor[] = prim::ListConstruct(%ret.1, %57)
#   return (%tensors.1)
# After:
#   ...
#   %57 : Float(1, 32, strides=[32, 1], requires_grad=0, device=cpu) = prepacked::linear_clamp_run(%56, %28)
#   %ret.1 : Float(1, 12, strides=[12, 1], requires_grad=0, device=cpu) = aten::softmax(%59, %9, %30)
#   return (%ret.1, %57)
torch._C._te.replace_list_output_with_tuple(g)
torch._C._jit_pass_lower_all_tuples(g)

# Print the final graph
print(g)

# Now, compile it with NNC!
kernel = torch._C._te.TensorExprKernel(g)
print("SUCCESS!")
print(kernel.get_code_text("asm"))

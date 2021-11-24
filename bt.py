import torch
import sys

model = torch.load("bt.pt")

ii = model._generate_bundled_inputs_for_forward()
print([x.dtype for x in list(*ii)])
print([x.size() for x in list(*ii)])

model.eval()
model = torch.jit.freeze(model)
model._c.dump(attrs=False, params=False)

def prepare_graph(g, inputs, dynamic_dims):
  # Before:
  # graph(%self : __torch__.___torch_mangle_149.ModelProd, %bytes.1 : Tensor, %lens : Tensor):
  #   ...
  # After:
  # graph(%bytes.1 : Tensor, %lens : Tensor)
  #   ...
  torch._C._te.remove_unused_self_argument(g)

  # Inject shape/dtype/device info into the graph
  g = torch._C._jit_trace_graph(g, tuple(*inputs))
  torch._C._te.annotate_input_shapes(g, list(*inputs))

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

  # Replace dimensions from 'dynamic_dims' with symbolic shapes
  # Before:
  #   graph(%x : Long(10, 20, 30, 40))
  #     ...
  # After make_shapes_symbolic(graph, [10, 30]):
  #   graph(%x : Long(SS(-1), 20, SS(-2), 40))
  #     ...
  sym_indices = torch._C._te.make_shapes_symbolic(g, dynamic_dims)

  # Print the final graph
  print(g)
  return g, sym_indices

g = model.graph
g, sym_indices = prepare_graph(g, ii, [115])

print(sym_indices)
print(g)

# Now, compile it with NNC!
# If 'aot' was passed as an argument, just produce asm without running it.
if len(sys.argv) > 1 and sys.argv[1] == 'aot':
  # Now produce asm for various archs
  torch._C._te.set_llvm_aot_workflow(True)
  torch._C._te.set_llvm_target_triple("arm-linux")
  kernel = torch._C._te.TensorExprKernel(g, dict(), sym_indices)
  print(kernel.get_code_text("asm"))
  torch._C._te.set_llvm_target_triple("x86_64-linux")
  torch._C._te.set_llvm_target_cpu("haswell")
  kernel.recompile()
  print('===================================================================')
  print(kernel.get_codegen_stmt())
  print(kernel.get_code_text("asm"))
else:
  # Run NNC version and compare results vs reference
  kernel = torch._C._te.TensorExprKernel(g, dict(), sym_indices)
  extended_inputs = tuple(list(*ii) + [115])
  nnc_res = kernel.run(tuple(extended_inputs))
  ref_res = model(*tuple(*ii))[0]

  # Compare results with the reference
  for i in range(2):
    assert(torch.allclose(nnc_res[i], ref_res[i], rtol=1e-5, atol=1e-5))
  print("SUCCESS!")

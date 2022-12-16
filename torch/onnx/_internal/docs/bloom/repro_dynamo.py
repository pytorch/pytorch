import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sentence = "Question: Can I run BLOOM on a single GPU? Answer:"

# Load model
def load_model(model_name: str = "bigscience/bloom-560m"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_state_dict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentence, return_tensors="pt").to(0)
    print(inputs.keys())
    return model, inputs, tokenizer


# Inference in PyTorch
def run_model(model, inputs, tokenizer):
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)

    token_id = outputs[0][0][-1].argmax()
    answer = tokenizer.decode([token_id])
    print(f"{sentence}\n{answer}")


# Inference in dynamo
def run_dynamo(model, inputs, tokenizer):
    from torch import _dynamo as torchdynamo
    opt_model = torchdynamo.optimize("eager")(model)
    run_model(opt_model, inputs, tokenizer)


model, inputs, tokenizer = load_model()
run_model(model, inputs, tokenizer)  # this works
run_dynamo(model, inputs, tokenizer)  # this fails



# appendix
# Full error log
"""
Traceback (most recent call last):
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 1092, in run_node
    return nnmodule(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/nn/modules/module.py", line 1482, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/bowbao/stable_diffusion/lib/python3.8/site-packages/accelerate/hooks.py", line 156, in new_forward
    output = old_forward(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/nn/modules/sparse.py", line 162, in forward
    return F.embedding(
  File "/home/bowbao/pytorch/torch/nn/functional.py", line 2210, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
  File "/home/bowbao/pytorch/torch/_subclasses/fake_tensor.py", line 812, in __torch_dispatch__
    raise Exception(
Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. Please convert all Tensors to FakeTensors first. Found in aten.embedding.default(*(Parameter containing:
tensor([[-0.0099, -0.0048, -0.0111,  ..., -0.0426,  0.0099,  0.0212],
        [ 0.0048, -0.0127,  0.0138,  ..., -0.0448,  0.0003, -0.0120],
        [ 0.0065,  0.0239,  0.0050,  ..., -0.0431, -0.0067,  0.0137],
        ...,
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012],
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012],
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012]],
       device='cuda:3', dtype=torch.float16, requires_grad=True), FakeTensor(FakeTensor(..., device='meta', size=(1, 14), dtype=torch.int64), cuda:3)), **{})

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 1046, in get_fake_value
    return wrap_fake_exception(
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 712, in wrap_fake_exception
    return fn()
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 1047, in <lambda>
    lambda: run_node(tx.output, node, args, kwargs, nnmodule)
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 1096, in run_node
    raise RuntimeError(
RuntimeError: Failed running call_module self_transformer_word_embeddings(*(FakeTensor(FakeTensor(..., device='meta', size=(1, 14), dtype=torch.int64), cuda:3),), **{}):
Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. Please convert all Tensors to FakeTensors first. Found in aten.embedding.default(*(Parameter containing:
tensor([[-0.0099, -0.0048, -0.0111,  ..., -0.0426,  0.0099,  0.0212],
        [ 0.0048, -0.0127,  0.0138,  ..., -0.0448,  0.0003, -0.0120],
        [ 0.0065,  0.0239,  0.0050,  ..., -0.0431, -0.0067,  0.0137],
        ...,
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012],
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012],
        [-0.0028, -0.0038, -0.0012,  ..., -0.0252,  0.0013,  0.0012]],
       device='cuda:3', dtype=torch.float16, requires_grad=True), FakeTensor(FakeTensor(..., device='meta', size=(1, 14), dtype=torch.int64), cuda:3)), **{})
(scroll up for backtrace)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "repro_dynamo.py", line 56, in <module>
    run_dynamo(model, inputs, tokenizer)
  File "repro_dynamo.py", line 41, in run_dynamo
    run_model(opt_model, inputs, tokenizer)
  File "repro_dynamo.py", line 30, in run_model
    outputs = model(**inputs, return_dict=False)
  File "/home/bowbao/pytorch/torch/nn/modules/module.py", line 1482, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/eval_frame.py", line 82, in forward
    return self.dynamo_ctx(self._orig_mod.forward)(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/eval_frame.py", line 211, in _fn
    return fn(*args, **kwargs)
  File "/home/bowbao/stable_diffusion/lib/python3.8/site-packages/accelerate/hooks.py", line 151, in new_forward
    args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
  File "/home/bowbao/stable_diffusion/lib/python3.8/site-packages/accelerate/hooks.py", line 156, in <graph break in new_forward>
    output = old_forward(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/eval_frame.py", line 332, in catch_errors
    return callback(frame, cache_size, hooks)
  File "/home/bowbao/pytorch/torch/_dynamo/convert_frame.py", line 479, in _convert_frame
    result = inner_convert(frame, cache_size, hooks)
  File "/home/bowbao/pytorch/torch/_dynamo/convert_frame.py", line 103, in _fn
    return fn(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 90, in time_wrapper
    r = func(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/convert_frame.py", line 339, in _convert_frame_assert
    return _compile(
  File "/home/bowbao/pytorch/torch/_dynamo/convert_frame.py", line 398, in _compile
    out_code = transform_code_object(code, transform)
  File "/home/bowbao/pytorch/torch/_dynamo/bytecode_transformation.py", line 341, in transform_code_object
    transformations(instructions, code_options)
  File "/home/bowbao/pytorch/torch/_dynamo/convert_frame.py", line 385, in transform
    tracer.run()
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 1686, in run
    super().run()
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 537, in run
    and self.step()
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 500, in step
    getattr(self, inst.opname)(inst)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 306, in wrapper
    return inner_fn(self, inst)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 1014, in CALL_FUNCTION_KW
    self.call_function(fn, args, kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 434, in call_function
    self.push(fn.call_function(self, args, kwargs))
  File "/home/bowbao/pytorch/torch/_dynamo/variables/nn_module.py", line 220, in call_function
    return tx.inline_user_function_return(
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 470, in inline_user_function_return
    result = InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 1764, in inline_call
    return cls.inline_call_(parent, func, args, kwargs)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 1819, in inline_call_
    tracer.run()
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 537, in run
    and self.step()
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 500, in step
    getattr(self, inst.opname)(inst)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 306, in wrapper
    return inner_fn(self, inst)
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 965, in CALL_FUNCTION
    self.call_function(fn, args, {})
  File "/home/bowbao/pytorch/torch/_dynamo/symbolic_convert.py", line 434, in call_function
    self.push(fn.call_function(self, args, kwargs))
  File "/home/bowbao/pytorch/torch/_dynamo/variables/nn_module.py", line 201, in call_function
    return wrap_fx_proxy(
  File "/home/bowbao/pytorch/torch/_dynamo/variables/builder.py", line 731, in wrap_fx_proxy
    return wrap_fx_proxy_cls(
  File "/home/bowbao/pytorch/torch/_dynamo/variables/builder.py", line 768, in wrap_fx_proxy_cls
    example_value = get_fake_value(proxy.node, tx)
  File "/home/bowbao/pytorch/torch/_dynamo/utils.py", line 1066, in get_fake_value
    raise TorchRuntimeError() from e
torch._dynamo.exc.TorchRuntimeError:

from user code:
   File "/home/bowbao/transformers/src/transformers/models/bloom/modeling_bloom.py", line 903, in forward
    transformer_outputs = self.transformer(
  File "/home/bowbao/transformers/src/transformers/models/bloom/modeling_bloom.py", line 729, in forward
    inputs_embeds = self.word_embeddings(input_ids)

Set torch._dynamo.config.verbose=True for more information


You can suppress this exception and fall back to eager by setting:
    torch._dynamo.config.suppress_errors = True

"""

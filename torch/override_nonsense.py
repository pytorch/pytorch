import torch
import re


def extract_int_value(v):
    return int(re.search(r"\[value={(.*)}]", str(v)).group(1))


def extract_string_value(v):
    return re.search(r"\[value=\"(.*)\"]", str(v)).group(1)


def extract_int_array_value(v):
    digits = re.search(r"value=([\s\d]+)", str(v)).group(1)
    numbers = [x for x in digits.split(" ") if x]
    return [int(i) for i in numbers]


def slice_backward_symbolic(g, out_grad, input_sizes, dim, start, end, step):

    # Extract the slice axis from the string serialization of "dim". We'll
    # assume this value isn't dynamic...
    # (there must be a better way to do this...)
    dimval = extract_int_value(dim)
    in_static_shape = extract_int_array_value(input_sizes)
    in_size_at_dim = in_static_shape[dimval]

    # build input gradient shape, which is basically the shape of the output
    # gradient overridden at the "dim" position:
    in_grad_shape = g.op(
        "ScatterElements",
        g.op("Shape", out_grad),
        g.op("Constant", value_t=torch.tensor([dimval])),
        g.op("Constant", value_t=torch.tensor([in_size_at_dim]))
    )

    # A tensor full of zeros that we're going to insert the gradients into.
    in_grad = g.op("ConstantOfShape", in_grad_shape)
    
    # We're basically going to do in_grad[..., start:end:step, ...] = out_grad
    # where the slicing happens on the "dim" axis:

    # Extract the start/end/step values in the same (stupid) way, attempting to
    # handle "until the end of the array" and negative values correctly...
    # This assumes we're slicing on a non dynamic axis, which is probably
    # fairly reasonable because why would you slice along eg the batch
    # dimension you absolute barbarian?

    startval = extract_int_value(start)
    if startval < 0:
        startval += in_size_at_dim

    endval = min(extract_int_value(end), in_size_at_dim)
    if endval < 0:
        endval += in_size_at_dim
    stepval = extract_int_value(step)

    inds = range(startval, endval, stepval)
    indices = g.op("Constant", value_t=torch.tensor(inds))

    # Now we need to reshape indices into shape [1,1,...,n,...,1,1], where
    # n = len(inds) lives on the "dim" axis, then expand so it's got the
    # same shape as out_grad (let's not just do one reshape in case some
    # of the axes are dynamic):
    padded_shape = [1] * len(in_static_shape)
    padded_shape[dimval] = len(inds)
    padded_shape = g.op("Constant", value_t=torch.tensor(padded_shape))
    indices = g.op("Reshape", indices, padded_shape)
    indices = g.op("Expand", indices, g.op("Shape", out_grad))

    # goddammit, they don't make this incredibly straightforward looking
    # operation easy do they...
    grad_input = g.op("ScatterElements", in_grad, indices, out_grad, axis_i=dimval)
    return grad_input


# Register the symbolic function for the specific op and version
torch.onnx.register_custom_op_symbolic('aten::slice_backward', slice_backward_symbolic, 17)


def select_backward_symbolic(g, out_grad, input_sizes, dim, index):

    # we can't use input_sizes for the shape of the input gradient, as that
    # bakes in the value we give it at onnx export time and means the onnx
    # doesn't work with dynamic shapes. We have to build it from the output
    # gradient by inserting the size at the selected dimension instead.
    dimval = extract_int_value(dim)
    in_static_shape = extract_int_array_value(input_sizes)

    in_grad_shape = g.op("ConcatFromSequence", g.op(
        "SequenceInsert",
        g.op("SplitToSequence", g.op("Shape", out_grad), axis_i=0, keepdims_i=1),
        g.op("Constant", value_t=torch.tensor([in_static_shape[dimval]])),
        dim
    ), axis_i=0)

    # A tensor full of zeros that we're going to insert the gradients into.
    in_grad = g.op("ConstantOfShape", in_grad_shape)
    
    # We're basically going to do in_grad[..., index, ...] = out_grad
    # where the indexing happens on the "dim" axis.

    # Add an extra dimension of size 1 to out_grad in the "dim" position.
    # Basically do out_grad = out_grad[...,None,...]
    out_grad_extraaxis_shape = g.op("ConcatFromSequence", g.op(
        "SequenceInsert",
        g.op("SplitToSequence", g.op("Shape", out_grad), axis_i=0, keepdims_i=1),
        g.op("Constant", value_t=torch.tensor([1])),
        dim
    ), axis_i=0)
    out_grad_extraaxis = g.op("Reshape", out_grad, out_grad_extraaxis_shape)
    
    # Create a tensor with the same shape, filled with the value "index"
    # so we can use a ScatterElements node:
    padded_shape = [1] * len(in_static_shape)
    padded_shape = g.op("Constant", value_t=torch.tensor(padded_shape))
    indices = g.op("Reshape", index, padded_shape)
    indices = g.op("Expand", indices, out_grad_extraaxis_shape)

    # ok, assume the value of "dim" is hard coded/not dynamic:
    grad_input = g.op("ScatterElements", in_grad, indices, out_grad_extraaxis, axis_i=dimval)
    return grad_input


# Register the symbolic function for the specific op and version
torch.onnx.register_custom_op_symbolic('aten::select_backward', select_backward_symbolic, 17)


def _index_put_impl_symbolic(g, dst, indices, values, accumulate, unsafe):

    return torch.onnx.symbolic_opset11.index_put(g, dst, indices, values, accumulate)


torch.onnx.register_custom_op_symbolic('aten::_index_put_impl', _index_put_impl_symbolic, 17)


def new_empty_strided_symbolic(g, *args):

    #print("new_empty_strided!!!")
    #for a in args:
    #    print(a)
    
    return g.op("ConstantOfShape", args[1])


torch.onnx.register_custom_op_symbolic('aten::new_empty_strided', new_empty_strided_symbolic, 17)


def copy_symbolic(g, *args):
    return args[0]


torch.onnx.register_custom_op_symbolic('aten::copy', copy_symbolic, 17)


def gelu_backward_symbolic(g, out_grad, input, approximate):

    approximate_val = extract_string_value(approximate)
    if approximate_val != "none":
        raise RuntimeError("I'm lazy and not supporting tanh approximations. Do it yerself!")

    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    kAlpha = g.op("Constant", value_t=torch.tensor(M_SQRT1_2))
    kBeta = g.op("Constant", value_t=torch.tensor(M_2_SQRTPI * M_SQRT1_2 * 0.5))

    # cdf = 0.5 * (1 + erf(input * kAlpha))
    cdf = g.op(
        "Mul",
        g.op(
            "Add",
            g.op("Erf", g.op("Mul", input, kAlpha)),
            g.op("Constant", value_t=torch.tensor(1.0))
        ),
        g.op("Constant", value_t=torch.tensor(0.5))
    )

    # pdf = kBeta * exp(input * input * -0.5)
    input_squared = g.op("Mul", input, input)
    exparg = g.op(
        "Mul",
        input_squared,
        g.op("Constant", value_t=torch.tensor(-0.5))
    )
    pdf = g.op(
        "Mul",
        g.op("Exp", exparg),
        kBeta
    )

    # return out_grad * (cdf + input * pdf)
    return g.op(
        "Mul",
        out_grad,
        g.op(
            "Add",
            cdf,
            g.op("Mul", input, pdf)
        )
    )


torch.onnx.register_custom_op_symbolic('aten::gelu_backward', gelu_backward_symbolic, 17)

def _softmax_backward_data_symbolic(g, out_grad, output, dim, half_to_float):
    
    new_grad_output = g.op(
        "Mul",
        out_grad,
        output
    )
    
    dimconst = g.op(
        "Expand",
        dim,
        g.op(
            "Constant",
            value_t = torch.tensor([1])
        )
    )

    grad_input = g.op(
        "Sub",
        new_grad_output,
        g.op(
            "Mul",
            output,
            g.op(
                "ReduceSum",
                new_grad_output, dimconst
            )
        )
    )
    return grad_input

torch.onnx.register_custom_op_symbolic('aten::_softmax_backward_data', _softmax_backward_data_symbolic, 17)

def layer_norm(x, normalized_shape, weight, bias, eps, unused):
    # Assuming `normalized_shape` is the last 'n' dimensions of `x`
    # Compute the mean and variance of `x` along the last 'n' dimensions

    mean = x.sum(dim=(-1,), keepdim=True) / x.shape[-1]
    x_normalized = x - mean
    var = (x_normalized ** 2).sum(dim=(-1,), keepdim=True) / x.shape[-1]
    
    # Normalize `x` using the computed mean and variance
    # Is torch.sqrt() treated as a constant then? why??
    x_normalized = x_normalized / torch.sqrt(var + eps)
    
    # Apply the learnable weights and biases, if provided
    if weight is not None:
        x_normalized = x_normalized * weight
    if bias is not None:
        x_normalized = x_normalized + bias
    
    return x_normalized


torch.layer_norm = layer_norm


def scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal):

    # Calculate the dot product between queries and keys
    d_k = k.size(-1)  # Get the size of the key dimension

    q_batchonly = q.reshape(-1,q.shape[-2],q.shape[-1])
    k_batchonly = k.reshape(-1,k.shape[-2],k.shape[-1])
    v_batchonly = v.reshape(-1,v.shape[-2],v.shape[-1])

    leading_dims = q.shape[:-2]

    attn_logits_batchonly = torch.bmm(q_batchonly.contiguous(), k_batchonly.transpose(-2, -1).contiguous()) / (d_k ** 0.5)

    # Apply the attention mask (if provided)
    if attn_mask is not None:
        attn_logits_batchonly = attn_logits.masked_fill(attn_mask.reshape(-1,attn_mask.shape[-2],attn_mask.shape[-1]) == 0, float('-inf'))

    # If causal, mask out future positions (upper triangular part of the matrix)
    if is_causal:
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(attn_logits.device)
        attn_logits_batchonly = attn_logits_batchonly.masked_fill(causal_mask == 1, float('-inf'))
    

    # Apply softmax to get the attention weights
    attn_weights_batchonly = torch.nn.functional.softmax(attn_logits_batchonly, dim=-1)

    results = torch.bmm(attn_weights_batchonly, v_batchonly)
    return results.reshape( leading_dims + results.shape[-2:] )


torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention

import torch
import re
from torch.onnx import symbolic_opset11, symbolic_helper

def extract_int_value(v):
    return int(re.search(r"\[value={(.*)}]", str(v)).group(1))


def extract_string_value(v):
    return re.search(r"\[value=\"(.*)\"]", str(v)).group(1)


def extract_int_array_value(v):
    digits = re.search(r"value=([\s\d]+)", str(v)).group(1)
    numbers = [x for x in digits.split(" ") if x]
    return [int(i) for i in numbers]


def extract_list_len(v):
    items = re.search(r"ListConstruct\(([^\()]+)\)", str(v)).group(1)
    items = items.split(", ")
    return len(items)


def slice_backward_symbolic(g, out_grad, input, dim, start, end, step):

    # A tensor full of zeros that we're going to insert the gradients into.
    input_shape = g.op("Shape", input)
    in_grad = g.op("ConstantOfShape", input_shape)

    # Now we need an array mapping indices in the output
    # tensor to indices in the input, along the slicing axis:
    input_dim_size = g.op("Gather", input_shape, dim)

    zero = g.op("Constant", value_t=torch.tensor([0]))
    one = g.op("Constant", value_t=torch.tensor([1]))

    # All input inds:
    input_dim_inds = g.op(
        "Range",
        zero,
        input_dim_size,
        one
    )

    input_dim_inds = g.op(
        "Slice",
        input_dim_inds,
        g.op("Reshape", start, one),
        g.op("Reshape", end, one),
        g.op("Constant", value_t=torch.tensor([0])),
        g.op("Reshape", step, one),
    )

    # we have to reshape/expand this now:
    dimval = extract_int_value(dim)
    padded_shape = g.op("ConstantOfShape", g.op("Shape", input_shape), value_t=torch.tensor([1], dtype=torch.int64))
    padded_shape = g.op(
        "ScatterElements",
        padded_shape,
        g.op("Reshape", dim, one),
        g.op("Reshape", g.op("Shape", input_dim_inds), one)
    )
    input_dim_inds = g.op("Reshape", input_dim_inds, padded_shape)
    input_dim_inds = g.op("Expand", input_dim_inds, g.op("Shape", out_grad))

    # finally use this array to scatter the gradients into the input grad:
    return g.op("ScatterElements", in_grad, input_dim_inds, out_grad, axis_i=dimval)


# Register the symbolic function for the specific op and version
torch.onnx.register_custom_op_symbolic('aten::slice_backward', slice_backward_symbolic, 17)


def select_backward_symbolic(g, out_grad, input, dim, index):


    # A tensor full of zeros that we're going to insert the gradients into.
    input_shape = g.op("Shape", input)
    in_grad = g.op("ConstantOfShape", input_shape)
    
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
    padded_shape = g.op("ConstantOfShape", g.op("Shape", input_shape), value_t=torch.tensor([1], dtype=torch.int64))
    indices = g.op("Reshape", index, padded_shape)
    indices = g.op("Expand", indices, out_grad_extraaxis_shape)

    # ok, assume the value of "dim" is hard coded/not dynamic:
    dimval = extract_int_value(dim)
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


def _cat_backward_symbolic(g, out_grad, inputs, dim):

    num_inputs = extract_list_len(inputs)

    in_grads = []
    one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
    start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
    for i in range(num_inputs):
        
        input = g.op(
            "SequenceAt",
            inputs,
            g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
        )

        input_shape = g.op("Shape", input)
        input_dim_size = g.op("Gather", input_shape, dim)

        end_idx = g.op("Add", start_idx, input_dim_size)

        in_grads.append(
            g.op(
                "Slice",
                out_grad,
                start_idx,
                end_idx,
                g.op("Reshape", dim, one)
            )
        )

        start_idx = end_idx

    return g.op("SequenceConstruct",*in_grads)

torch.onnx.register_custom_op_symbolic('aten::cat_backward', _cat_backward_symbolic, 17)


def _squeeze_symbolic(g, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    # dim as a tensor
    if not symbolic_helper._is_constant(dim):
        return symbolic_helper._squeeze_helper(g, self, [dim])

    dim = symbolic_helper._get_const(dim, "i", "dim")
    return symbolic_helper._squeeze_helper(g, self, [dim])

torch.onnx.register_custom_op_symbolic('aten::squeeze', _squeeze_symbolic, 17)


def index_backward_native_symbolic(g, grad, self, indices):

    shape = g.op(
        "Shape",
        self
    )
    in_grad = g.op(
        "ConstantOfShape",
        shape
    )

    return symbolic_opset11.index_put(g, in_grad, indices, grad, True)

torch.onnx.register_custom_op_symbolic('aten::index_backward_native', index_backward_native_symbolic, 17)

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

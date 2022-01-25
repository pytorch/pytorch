"""
In this exercise, we will look into loop level statement more closely
using NNC. We will use NNC IR to construct a conv2d layer.
"""

import torch
import torch._C._te as te

#########################################################
########### Reference Conv2d ############################
#########################################################

# First, we create a reference conv2d layer using the following arguments.
conv_args = (16, 16, (3, 3), (2, 2), (1, 1), (1, 1), 16, False, 'zeros')
input_shape = (1, 16, 56, 56)
conv2d_ref = torch.nn.Conv2d(*conv_args)

# We create the input image and feed it to the reference conv2d layer to obtain the output shape.
image = torch.rand(*input_shape)
weight = conv2d_ref.weight
with torch.no_grad():
    out_ref = conv2d_ref(image)

out_shape = out_ref.shape
print("Output shape: ", out_shape)
# Output shape: (1, 16, 28, 28)

#########################################################
########### Create Conv2d using NNC #####################
#########################################################

# First, we create a set of NNC variables as iterators for conv2d computation:
# batch size iterator n, group iterator g, k iterator, image height iterator
# h, image width iterator w
(n, g, k, h, w) = [te.VarHandle(x, torch.int32) for x in ["n", "g", "k", "h", "w"]]
# continuing iterators: weight channel iterator c, weight height iterator r,
# weight width iterator s
(c, r, s) = [te.VarHandle(x, torch.int32) for x in ["c", "r", "s"]]

# We then create a set of expressions that holds the upper bounds in the above
# iteration dimensions
N, H, W, C, Ho, Wo, Co, R, S, groups = 1, 56, 56, 16, 28, 28, 16, 3, 3, 16
CC, KK, NN, GG, HH, WW, SS, RR = [te.ExprHandle.int(x) for x in [C//groups, C//groups, N, groups, Ho, Wo, S, R]]

# We create expressions for specific arguments: strides, padding
St, P = 2, 1
ST, PP = [te.ExprHandle.int(x) for x in [St, P]]

# Second, we create buffers for input image, weights, and output
dtype = te.Dtype.Float
Pimage = te.BufHandle("image", [te.ExprHandle.int(x) for x in [N, C, H, W]], dtype)
Pweight = te.BufHandle("weight", [te.ExprHandle.int(x) for x in [Co, C//groups, R, S]], dtype)
Pconv = te.BufHandle("conv", [te.ExprHandle.int(x) for x in [N, Co, Ho, Wo]], dtype)

# Now we can define how to compute conv2d based on input image and weights
in_chan = c + g * CC
out_chan = k + g * KK
h_image = h * ST + r - PP
w_image = w * ST + s - PP
# Construct the store stmt that multiplies weights to the input image, and adds it to the output
stmt = Pconv.store([n, out_chan, h, w], Pconv.load([n, out_chan, h, w]) + Pimage.load([n, in_chan, h_image, w_image]) * Pweight.load([out_chan, c, r, s]))
# Construct the conditional stmt that skips padding areas
cond = (h_image >= te.ExprHandle.int(0)) & (h_image < te.ExprHandle.int(H))
cond = (w_image >= te.ExprHandle.int(0)) & (w_image < te.ExprHandle.int(W)) & cond
stmt = te.Cond.make(cond, stmt, None)
# Construct the final nested loop, and print it out!
for v, bound in zip([s, r, c, w, h, k, g, n], [SS, RR, CC, WW, HH, KK, GG, NN]):
    stmt = te.For.make(v, te.ExprHandle.int(0), bound, stmt)
print("Conv2d Stmt: ", stmt)
"""
Conv2d Stmt:  for (int n = 0; n < 1; n++) {
  for (int g = 0; g < 16; g++) {
    for (int k = 0; k < 1; k++) {
      for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
          for (int c = 0; c < 1; c++) {
            for (int r = 0; r < 3; r++) {
              for (int s = 0; s < 3; s++) {
                if ((((w * 2 + s) - 1>=0 ? 1 : 0) & ((w * 2 + s) - 1<56 ? 1 : 0)) & (((h * 2 + r) - 1>=0 ? 1 : 0) & ((h * 2 + r) - 1<56 ? 1 : 0))) {
                  conv[n, k + g * 1, h, w] = (conv[n, k + g * 1, h, w]) + (image[n, c + g * 1, (h * 2 + r) - 1, (w * 2 + s) - 1]) * (weight[k + g * 1, c, r, s]);
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

# Next, feed the constructed stmt to loopnest to prepare for codegen
loopnest = te.LoopNest(stmt, [Pconv])
loopnest.prepare_for_codegen()
stmt = te.simplify(loopnest.root_stmt())

# Finally, we can run the stmt with the same inputs. Let's compare its results
# to the reference conv2d we created in section 1.
LLVM_ENABLED = torch._C._llvm_enabled()
if LLVM_ENABLED:
    codegen = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [Pimage, Pweight, Pconv]])
    out = torch.zeros(out_shape)
    codegen.call([image, weight, out])
    torch.testing.assert_allclose(out, out_ref)
    # As expected, the outputs are the same!

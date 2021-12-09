import torch.nn as nn
import torch

encoder_layer = nn.TransformerEncoderLayer(4, 2, 8, 0, batch_first=False)
src = torch.rand(3, 4)
out = encoder_layer(src)

# for i in range(10):
#     out_no_batch = encoder_layer(src[i])
#     torch.testing.assert_allclose(out[i], out_no_batch)

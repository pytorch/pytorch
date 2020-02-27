import torch
import torch.nn.functional as F

inv = torch.zeros((3, 8), dtype=torch.float).requires_grad_()
indices = torch.zeros((2, 3), dtype=torch.long)

comb = torch.sparse.FloatTensor(indices, inv, (4, 4, 8)).to_dense()
big = F.pad(comb, (0, 0, 1, 1, 1, 1))
shaped = big.view(-1, 8).permute(1, 0).unsqueeze(0)
res = F.fold(shaped, output_size=(5, 5),
             kernel_size=(2, 2), padding=(1, 1))
res.sum().backward()

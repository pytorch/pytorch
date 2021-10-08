
python -c "import torch; from torch import nn; m = torch.jit.load('/home/ivankobzarev/rsync/nnc/mnetv3-large/mobilenet_v3_large_opt.pt').eval(); inputs = list(m.graph.inputs()); size = [1, 3, 224, 224]; inputs[1].setType(inputs[1].type().with_sizes(size));print(m.graph)"

import torch.nn as nn

avg_pool1 = nn.AdaptiveAvgPool2d((1, None))
avg_pool2 = nn.AdaptiveAvgPool2d((None, 1))
max_pool1 = nn.AdaptiveMaxPool2d((1, None))
max_pool2 = nn.AdaptiveMaxPool2d((None, 1))

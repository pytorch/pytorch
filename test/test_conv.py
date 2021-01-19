import torch
from torch import nn
# with torch.no_grad():
class Conv2dTest(nn.Module):

    def __init__(self) -> None:
        super(Conv2dTest, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        y = self.conv(x)
        z = y.relu_()
        return z



conv_module = Conv2dTest()
scripted = torch.jit.script(conv_module)

x = torch.rand(128, 3, 227, 227)
r = conv_module(x)
scripted(x)
sr = scripted(x)
torch.allclose(r, sr)
print("Done!")

            # class Test(torch.nn.Module):
            #     def __init__(self):
            #         super(Test, self).__init__()
            #         self.conv = torch.nn.Conv2d(1, 20, 5, 1)
            #         self.bn = torch.nn.BatchNorm2d(num_features=20)

            #     def forward(self, x):
            #         x = self.conv(x)
            #         x = self.bn(x)
            #         return x
            # m = torch.jit.script(Test())
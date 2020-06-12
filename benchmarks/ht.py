import torch
import torch.nn as nn
import time
from executor import set_mode

class custom_rnn(torch.jit.ScriptModule):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        nonlinearity="relu",
        device="cuda",
    ):

        super(custom_rnn, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.device = device

        self.w = nn.Linear(
            self.input_size, 2 * self.hidden_size, bias=False
        ).to(device)
        
        # Initilizing initial state h
        self.h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=self.device,
            )

    @torch.jit.script_method
    def forward(self, x):
        ht = self.h_init
        
        # Loop over time axis
        for k in range(x.shape[1]):
            ht = ht + 1.0
        return ht

    
if __name__ == "__main__":
    set_mode()

    inp_tensor = torch.rand([4, 500, 40]).to('cuda')
    net = custom_rnn(40, 512, 1, 4, device='cuda').to('cuda')
    start = time.time()

    for i in range(1000):
        out_tensor = net(inp_tensor)

    end = time.time()
    print(end - start)


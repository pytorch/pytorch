import torch
from torch import nn


class GRU(nn.Module):
    ''' GRU
    Reference
    https://discuss.pytorch.org/t/implementation-of-multiplicative-lstm/2328/9
    https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py#L46
    '''

    def __init__(self, input_size, hidden_size, seq_len, batch_first=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.seq_len = seq_len

        self.input_weights = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 3 * hidden_size)

    def step(self, input, hidden):
        hx = hidden
        w_ih = self.input_weights(input)
        w_hh = self.hidden_weights(hx)

        i_r, i_c, i_u = w_ih.chunk(3, 1)
        h_r, h_c, h_u = w_hh.chunk(3, 1)

        updategate = torch.sigmoid(i_u+h_u)
        resetgate = torch.sigmoid(i_r+h_r)

        new_state = torch.tanh(i_c + resetgate * h_c)

        hy = (1.-updategate)*hx + updategate*new_state

        return hy

    def forward(self, input, hidden):

        if self.batch_first:
            input = input.transpose(0, 1)

        # Main loop
        output = []
        for i in range(self.seq_len):
            hidden = self.step(input[i], hidden)
            output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


if __name__ == '__main__':
    input_size = 48
    hidden_size = 48
    batch_size = 20

    cuda_id = 0
    # cuda_id = None

    seq_len = 200
    # seq_len = 100

    gru = GRU(input_size, hidden_size, seq_len, batch_first=True)

    input = torch.rand(batch_size, seq_len, input_size)
    hidden = torch.rand(batch_size, hidden_size)

    if cuda_id is not None:
        torch.cuda.set_device(cuda_id)
        gru = gru.cuda()
        input = input.cuda()
        hidden = hidden.cuda()

    traced_gru = torch.jit.trace(gru, (input, hidden))


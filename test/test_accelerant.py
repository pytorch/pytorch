import torch
from torch import nn


class Accelerant:
    def __init__(self, scripted):
        # this is an nn.Module
        if hasattr(scripted, "_c"):
            scripted._c = torch._C._freeze_module(scripted._c)
            self.accelerant = torch._C._jit_to_accelerant(
                scripted._c, scripted._c._get_method("forward").graph
            )
        else:
            self.accelerant = torch._C._jit_to_accelerant(scripted.graph)

    def __call__(self, *inps):
        return self.accelerant.run(inps)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


def trivial_graph(a, b, c):
    s = torch.tensor([[3, 3], [3, 3]])
    return a + b * c + s


if __name__ == "__main__":
    HID_DIM = 256
    QUERY_LEN = 8
    BATCH_SIZE = 128
    LAYERS = 3
    HEADS = 8
    DROPOUT = 0.1
    device = torch.device("cpu")
    attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
    src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
    src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

    attention.eval()
    attention = torch.jit.script(attention)
    attention.eval()
    o_ref = attention(src, src, src, src_mask)

    attention_a = Accelerant(attention)
    o_test = attention_a(src, src, src, src_mask)
    for a, b in zip(o_ref, o_test):
        torch.testing.assert_allclose(a, b)

    s = torch.full((2, 2), 2)
    tg = torch.jit.script(trivial_graph)
    o_ref = tg(s, s, s)
    tg_a = Accelerant(tg)
    o_test = tg_a(s, s, s)[0]
    torch.testing.assert_allclose(o_ref, o_test)

import torch
import torch.nn as nn
from torch._higher_order_ops.scan import scan
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1)

LAYERS = 3
BATCH_SIZE = 2
SEQ_LEN = 5
FEATURE_DIM = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNNLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(FEATURE_DIM*2, FEATURE_DIM) for _ in range(LAYERS)])
        self.num_layers = LAYERS

    def forward(self, initial, inputs_sequence):
        B, T, _ = inputs_sequence.shape
        hs_list = initial
        all_out = []
        for t in range(T):
            input = inputs_sequence[:, t, :]
            for li, layer in enumerate(self.layers):
                input_concat = torch.cat((hs_list[li], input), dim=-1)
                update = layer(input_concat)
                hs_list[li] = hs_list[li] + update
                input = hs_list[li]

            all_out.append(input)

        return torch.stack(all_out, dim=1)

class RNNScanList(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(FEATURE_DIM*2, FEATURE_DIM) for _ in range(LAYERS)])
        self.num_layers = LAYERS

    def forward(self, initial, input_sequence):
        def step(carry, input):
            hs_list = carry[:]
            for li, layer in enumerate(self.layers):
                h_prev_li = hs_list[li]
                input_concat = torch.cat((h_prev_li, input), dim=-1)
                update = layer(input_concat)
                h_curr_li = h_prev_li + update
                hs_list[li] = h_curr_li
                input = h_curr_li
            return [t.clone() for t in hs_list], input.clone()

        _, all_outputs_scan = scan(step, initial, input_sequence, dim=1)
        return all_outputs_scan.transpose(0, 1)

class RNNScanTensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(FEATURE_DIM*2, FEATURE_DIM) for _ in range(LAYERS)])
        self.num_layers = LAYERS

    def forward(self, initial, input_sequence):
        def step(carry_tensor, xs_input):
            input = xs_input
            hs_tensor = carry_tensor
            for li, layer in enumerate(self.layers):
                current_h_prev_li_slice = hs_tensor[:, li, :]
                input_concat = torch.cat((current_h_prev_li_slice, input), dim=-1)
                update = layer(input_concat)
                h_curr_li = current_h_prev_li_slice + update
                hs_tensor = hs_tensor.clone()
                hs_tensor[:, li, :] = h_curr_li
                input = h_curr_li
            return hs_tensor.clone(), input.clone()

        hs_stacked = torch.stack(initial, dim=1)
        _, all_outputs_scan = scan(step, hs_stacked, input_sequence, dim=1)
        return all_outputs_scan.transpose(0, 1)



def run_test_and_print_grads(model, initial_hs, inputs, label):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    current_initial_hs = [h.detach().clone().requires_grad_(h.requires_grad) for h in initial_hs]
    current_inputs = inputs.detach().clone().requires_grad_(inputs.requires_grad)

    out = model(current_initial_hs, current_inputs)
    loss = out.sum()
    loss.backward()
    print(f'\n{label} -> loss: {loss.item():.6f}')
    for i, layer in enumerate(model.layers):
        print(f'Layer {i} grad: {layer.weight.grad.view(-1)[:5]} {layer.weight.grad.view(-1)[-5:]} {layer.weight.grad.sum().item():2f}')



torch.manual_seed(0)

initial_hs_template = [torch.zeros(BATCH_SIZE, FEATURE_DIM, requires_grad=True, dtype=torch.float32).to(DEVICE) for _ in range(LAYERS)]
inputs_template = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM, requires_grad=True, dtype=torch.float32).to(DEVICE)

print("--- UNCOMPILED ---")
model_sl_uc = RNNScanList().to(DEVICE)
run_test_and_print_grads(model_sl_uc, initial_hs_template, inputs_template, "ScanList Uncompiled")

model_st_uc = RNNScanTensor().to(DEVICE)
model_st_uc.load_state_dict(model_sl_uc.state_dict())
run_test_and_print_grads(model_st_uc, initial_hs_template, inputs_template, "ScanTensor Uncompiled")

model_loop_uc = RNNLoop().to(DEVICE)
model_loop_uc.load_state_dict(model_sl_uc.state_dict())
run_test_and_print_grads(model_loop_uc, initial_hs_template, inputs_template, "Loop Uncompiled")

print("\n--- COMPILED ---")

model_sl_to_compile = RNNScanList().to(DEVICE)
model_sl_to_compile.load_state_dict(model_sl_uc.state_dict())
compiled_model_sl = torch.compile(model_sl_to_compile)
run_test_and_print_grads(compiled_model_sl, initial_hs_template, inputs_template, "ScanList Compiled")

model_st_to_compile = RNNScanTensor().to(DEVICE)
model_st_to_compile.load_state_dict(model_sl_uc.state_dict())
compiled_model_st = torch.compile(model_st_to_compile)
run_test_and_print_grads(compiled_model_st, initial_hs_template, inputs_template, "ScanTensor Compiled")

model_loop_to_compile = RNNLoop().to(DEVICE)
model_loop_to_compile.load_state_dict(model_sl_uc.state_dict())
compiled_model_loop = torch.compile(model_loop_to_compile)
run_test_and_print_grads(compiled_model_loop, initial_hs_template, inputs_template, "Loop Compiled")

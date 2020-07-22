import torch

class adaroundObeserver(torch.quantization.observer.MinMaxObserver):
    def __init__(self):
        super(MinMaxObserver, self).__init__()
        self.V = torch.zeros(i,j)
        self.beta = 2
        self._lambda = .9

    def forward(self, x):
        pass

def loss_function(model, input):
    # model should contain its scaling parameters
    # its activation_post_process should have an observer with V

    scale = something
    weights = something
    beta = something
    _lambda = something
    V = something

    W_over_s = torch.floor_divide(weights, scale)
    W_plus_H = W_over_s + h_V(V)
    soft_quantized_weights = scale * torch.clamp(W_plus_H, 0, 255)
    soft_model = copy.deepcopy(model)
    soft_model.weights = soft_quantized_weights

    # Frobenius_norm = torch.norm(weights - soft_quantized_weights)
    Frobenius_norm = torch.norm(model.forward(input) - soft_model.forward(input))

    spreading_range = 2*V -1
    one_minus_beta = 1- (spreading_range ** beta)  # torch.exp
    f_reg = torch.sum(one_minus_beta)

    return Frobenius_norm + _lambda*f_reg

def h_V(V):
    sig_applied = torch.sigmoid(V)
    # scale_n_add = torch.add(torch.mul(V, 1.2), -0.1)
    scale_n_add = (V* 1.2) -0.1 #broadcast should work?
    clip = torch.clamp(scale_n_add, 0, 255)
    return clip


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()

    for image, target in data_loader:
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

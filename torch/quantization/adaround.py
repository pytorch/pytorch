import torch
from torch.quantization.fake_quantize import FakeQuantize

def clipped_sigmoid(continous_V):
    sigmoid_applied = torch.sigmoid(continous_V)
    scale_n_add = (continous_V* 1.2) -0.1  # broadcast should work?
    # TODO: dtypes
    if continous_V.dtype == torch.int8:
        clip = torch.clamp(scale_n_add, -128, 127)
    else:  # add other dtypes
        clip = torch.clamp(scale_n_add, -128, 127)
    return clip

def modified_quantized(weight, continous_V):
    W_over_s = torch.floor_divide(weight, scale)
    W_plus_H = W_over_s + clipped_sigmoid(continous_V)
    # TODO: dtypes
    if W.dype == torch.int8:
        soft_quantized_weights = scale * torch.clamp(W_plus_H, -128, 127)
    else:  # add dtype conditional for clambing range
        soft_quantized_weights = scale * torch.clamp(W_plus_H, -128, 127)
    return soft_quantized_weights

def loss_function(model, input):
    beta = 1  # something we get to play with
    _lambda = 1  # something we get to play with

    scale = something  # grab from the observer?
    weights = model.weights
    continous_V = model.weight_fake_quant.continous_V

    soft_model = copy.deepcopy(model)
    soft_model.weights = modified_quantized(weights, continous_V)

    # Frobenius_norm = torch.norm(weights - soft_quantized_weights)
    Frobenius_norm = torch.norm(model.forward(input) - soft_model.forward(input))

    spreading_range = 2 * continous_V - 1
    one_minus_beta = 1 - (spreading_range ** beta)  # torch.exp
    regulization = torch.sum(one_minus_beta)

    return Frobenius_norm + (_lambda * regulization)

class adaround(FakeQuantize):
    def __init__(self):
        super(FakeQuantize, self).__init__()
        self.continous_V = None

    def forward(self, x):
        # the x here is expected to be the parents? weights tensor
        if self.continous_V is None:
            self.continous_V = torch.zeros(x.size())  # random?
        return modified_quantized(x, self.continous_V)




araround_fake_quant = adaround.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
araround_qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            reduce_range=True),
                          weight=araround_fake_quant)

def main():
    prepared_model =
    prepared_model.train()
    img_data =
    optimizer = torch.optim.Adam([model.weight_fake_quant.continous_V], lr=0.001)

    for image, target in img_data:
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = loss_function(model, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()

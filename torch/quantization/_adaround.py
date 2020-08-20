import torch
import torch.nn as nn
import torch.nn.qat as nnqat

_supported_modules = {nn.Conv2d, nn.Linear}
_supported_modules_qat = {nnqat.Conv2d, nnqat.Linear}

def computeSqnr(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)

def get_module(model, name):
    ''' Given name of submodule, this function grabs the submodule from given model
    '''
    return dict(model.named_modules())[name]

def parent_child_names(name):
    '''Splits full name of submodule into parent submodule's full name and submodule's name
    '''
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return '', split_name[0]
    else:
        return split_name[0], split_name[1]

def get_param(module, attr):
    ''' Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    '''
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    else:
        return param

def learn_adaround(quantized_model, tuning_dataset, target_layers=None,
                   number_of_epochs=15, learning_rate=.25):
    ''' Inplace learning procedure for tuning the rounding scheme of the layers specified
    for the given model

    Args:
        quantized_model: a model with AdaRoundFakeQuantize modules attached to it
        tuning_dataset: a training/tuning dataset for the continous_V attribute in the fake_quants
        target_layers: the names of the layers that have an AdaRoundFakeQuantize module to train
        number_of_epochs: number of batches each layer is trained on
        learning_rate: learning rate of the training
    '''

    def optimize_V(leaf_module):
        '''Takes in a leaf module with an adaround attached to its
        weight_fake_quant attribute and runs an adam optimizer on the continous_V
        attribute on the adaround module
        '''
        leaf_module.weight_fake_quant.tuning = True

        def dummy_generator():
            yield leaf_module.weight_fake_quant.continous_V
        optimizer = torch.optim.Adam(dummy_generator(), lr=learning_rate)

        count = 0
        for data in tuning_dataset:
            output = quantized_model(data[0])
            loss = leaf_module.weight_fake_quant.layer_loss_function(count / number_of_epochs,
                                                                     leaf_module.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            if count == number_of_epochs:
                return

    if target_layers is None:
        target_layers = []
        for name, submodule in quantized_model.named_modules():
            if type(submodule) in _supported_modules_qat:
                target_layers.append(name)

    for layer_name in target_layers:
        layer = get_module(quantized_model, layer_name)
        optimize_V(layer)

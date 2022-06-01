import netron
from modelviewer import ModelViewer

# Provide path of trained model
model_path = 'cifar_net.pth'
model = ModelViewer.get_model(model_path)
layer_list, ref = ModelViewer.get_layers(model)
print(layer_list)
fig = ModelViewer.display_model(layer_list, ref)
ModelViewer.save_model(fig, model_path)
# compare against state-of-the-art third part library - "Netron" that is also used to visualize neural network models
netron.start(model_path)

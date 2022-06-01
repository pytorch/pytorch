# Author: Fiona Victoria Stanley Jothiraj

# The experimental implementation consists of adding components to the existing Pytorch’s architecture.
# The new implementation can help users to view the trained/inference models.
# Currently, no such support exists in PyTorch, and the users are forced to use other third-party tools from Tensorflow.
# However, integrating monitoring and visualization features like these would be of great use to the Pytorch developer community.

import torch
import matplotlib.pyplot as plt

class ModelViewer:
    @staticmethod
    def get_model(path):
        # load the pytorch model
        model = torch.load(path)
        return model

    @staticmethod
    def get_layers(model):
        # get the list of layers from model
        keysList = [key for key in model]
        # retreive the tensor size of each layer from the model
        # save as dictionary for faster access
        d = {}
        for i in keysList:
            tensor = model.get(i)
            tensor_shape = str(list(tensor.size())).replace(",", " ×")
            d[i] = tensor_shape
        # group similar layers - list of dictionary
        # ie. conv1.weight and conv1.bias are grouped together as one entity
        layer_list = []
        for i in d:
            name = i.rsplit('.', 1)[0]
            layer_name = [i for i in keysList if name in i]
            layers = {}
            for i in layer_name:
                layers[i] = d[i]
            layer_list.append(layers)
        # check to remove duplicate entries
        layer_list = [i for n, i in enumerate(layer_list) if i not in layer_list[n + 1:]]
        return layer_list, d

    @staticmethod
    def display_model(layers, ref):
        # generate rough sketch of architecture
        # plot size ranges from 0.0 to 1.0 in x and y axis
        fig, ax = plt.subplots(figsize=(5, 13))
        ax.axis("off")
        # initializing position of weights and bias components
        # initializers - Experimental version
        x, y = 0.1, 0.9
        step_size = 0.25
        print("\n\nGenerating architecture of trained model ...")
        for elem in layers:
            module = str(elem).replace('{', '').replace('}', '').replace("'", "").split(",")
            name = ""
            name += " Module\n"
            for i in module:
                name += "\n"
                name += str(i)
                name += "\n"
            ax.text(x, y, name, color='black', fontsize= 13, fontfamily="monospace",
                            bbox = dict(boxstyle="round", facecolor='none', edgecolor='black', ec=(0,0,0), fc=(0.949, 0.964, 0.917)))
            y = y - step_size
        plt.show()
        return fig

    @staticmethod
    def save_model(fig, img):
        # path to save the architecture diagram
        img = img.rsplit('.', 1)[0]
        # save the figure to file
        fig.savefig(img + ".png", bbox_inches='tight')
        # close the figure window
        plt.close(fig)

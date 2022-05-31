### Author: Fiona Victoria Stanley Jothiraj  ###

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
            tensor_shape = str(list(tensor.size())).replace(","," ×")
            d[i] = tensor_shape

        # group similar layers - list of dictionary
        # ie. conv1.weight and conv1.bias are grouped together as one entity
        layer_list = []
        for i in d:
            name = i.rsplit('.',1)[0]
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
        fig, ax = plt.subplots(figsize=(5, 13)) #plot size ranges from 0.0 to 1.0 in x and y axis
        ax.axis("off")

        # initializing position of weights and bias components   
        # initializers - Experimental version
        x, y = 0.1, 0.9
        step_size = 0.25

        print("\n\nGenerating architecture of trained model ...")
        for elem in layers:
            module = str(elem).replace('{', '').replace('}', '').replace("'", "").split(",")
            name = ""
            name+=" Module\n"
            for i in module:
                name+="\n"
                name+=str(i)
                name+="\n"
            ax.text(x, y,name,color='black',fontsize= 13, fontfamily="monospace",
                            bbox = dict(boxstyle="round", facecolor='none', edgecolor='black', ec=(0,0,0),
                            fc=(0.949, 0.964, 0.917)))
            y = y - step_size
        plt.show()
        return fig
                    
    def save_model(fig, img):
        img = img.rsplit('.',1)[0] # path to save the architecture diagram
        fig.savefig(img + ".png", bbox_inches='tight')   # save the figure to file
        plt.close(fig)    # close the figure window

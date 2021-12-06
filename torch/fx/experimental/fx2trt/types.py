from typing import Union, Callable, Dict, Tuple, Sequence

import tensorrt as trt
from torch.fx.node import Target, Argument

if hasattr(trt, "__version__"):
    TRTNetwork = trt.INetworkDefinition
    TRTTensor = trt.tensorrt.ITensor
    TRTLayer = trt.ILayer
    TRTPluginFieldCollection = trt.PluginFieldCollection
    TRTPlugin = trt.IPluginV2
    TRTDataType = trt.DataType
else:
    TRTNetwork = "trt.INetworkDefinition"
    TRTTensor = "trt.tensorrt.ITensor"
    TRTLayer = "trt.ILayer"
    TRTPluginFieldCollection = "trt.PluginFieldCollection"
    TRTPlugin = "trt.IPluginV2"
    TRTDataType = "trt.DataType"

Shape = Sequence[int]
ShapeRange = Tuple[Shape, Shape, Shape]

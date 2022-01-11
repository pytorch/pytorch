from typing import Tuple, Sequence

import tensorrt as trt

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

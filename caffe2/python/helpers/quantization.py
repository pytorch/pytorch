# @package quantization
# Module caffe2.python.helpers.quantization


def fused_8bit_rowwise_quantized_to_float(
    model, blob_in, blob_out
):
    """Fused8BitRowwiseQuantizedToFloat"""
    return model.net.Fused8BitRowwiseQuantizedToFloat(blob_in, blob_out)

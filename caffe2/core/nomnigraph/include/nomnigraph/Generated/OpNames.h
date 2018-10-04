
case NNKind::Relu:
  return "Relu";

case NNKind::Conv:
  return "Conv";

case NNKind::ConvRelu:
  return "ConvRelu";

case NNKind::ConvTranspose:
  return "ConvTranspose";

case NNKind::AveragePool:
  return "AveragePool";

case NNKind::AveragePoolRelu:
  return "AveragePoolRelu";

case NNKind::MaxPool:
  return "MaxPool";

case NNKind::MaxPoolRelu:
  return "MaxPoolRelu";

case NNKind::Sum:
  return "Sum";

case NNKind::SumRelu:
  return "SumRelu";

case NNKind::Send:
  return "Send";

case NNKind::Receive:
  return "Receive";

case NNKind::BatchNormalization:
  return "BatchNormalization";

case NNKind::Clip:
  return "Clip";

case NNKind::FC:
  return "FC";

case NNKind::GivenTensorFill:
  return "GivenTensorFill";

case NNKind::Concat:
  return "Concat";

case NNKind::Softmax:
  return "Softmax";

case NNKind::ChannelShuffle:
  return "ChannelShuffle";

case NNKind::Add:
  return "Add";

case NNKind::Reshape:
  return "Reshape";

case NNKind::Flatten:
  return "Flatten";

case NNKind::CopyToOpenCL:
  return "CopyToOpenCL";

case NNKind::CopyFromOpenCL:
  return "CopyFromOpenCL";

case NNKind::NCHW2NHWC:
  return "NCHW2NHWC";

case NNKind::NHWC2NCHW:
  return "NHWC2NCHW";

case NNKind::Declare:
  return "Declare";

case NNKind::Export:
  return "Export";

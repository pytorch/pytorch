import atexit
from caffe2.proto import caffe2_pb2
from google.protobuf.message import Message
from multiprocessing import Process
import os
import sys
import socket
from pycaffe2 import utils

try:
  from .libcaffe2_python import *
  has_gpu_support = True
except ImportError as e:
  print 'Pycaffe+GPU is not available. Using CPU only version.'
  from .libcaffe2_python_nogpu import *
  has_gpu_support = False
# We will always do a GlobalInit when we first import the workspace module.
# This is needed so we can make sure all underlying caffe2 stuff are properly
# initialized.
GlobalInit(sys.argv)
# libcaffe2_python contains a global Workspace that we need to properly delete
# when exiting. Otherwise, cudart will cause segfaults sometimes.
atexit.register(OnModuleExit)

try:
  import pycaffe2.mint.app
  _has_mint = True
except ImportError as err:
  print 'Mint is not available, possibly due to some downstream dependencies.'
  _has_mint = False

def _GetFreeFlaskPort():
  """Get a free flask port."""
  # We will prefer to use 5000. If not, we will then pick a random port.
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = sock.connect_ex(('127.0.0.1',5000))
  if result == 0:
    return 5000
  else:
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    # Race condition: between the interval we close the socket and actually
    # start a mint process, another process might have occupied the port. We
    # don't do much here as this is mostly for convenience in research rather
    # than 24x7 service.
    return port

def StartMint(root_folder=None, port=None):
  """Start a mint instance.

  TODO(Yangqing): this does not work well under ipython yet. According to
      https://github.com/ipython/ipython/issues/5862
  writing up some fix is a todo item.
  """
  if not _has_mint:
    print 'Mint is not available. Not starting the server.'
    return None
  if root_folder is None:
    root_folder = RootFolder()
  if port is None:
    port = _GetFreeFlaskPort()
  process = Process(target=pycaffe2.mint.app.main, args=(
      ['-p', str(port), '-r', root_folder],))
  process.start()
  print 'Mint running at http://{}:{}'.format(socket.getfqdn(), port)
  return process

def StringfyProto(obj):
  """Stringfy a protocol buffer object.

  Inputs:
    obj: a protocol buffer object, or a Pycaffe2 object that has a Proto()
        function.
  Outputs:
    string: the output protobuf string.
  Raises:
    AttributeError: if the passed in object does not have the right attribute.
  """
  if type(obj) is str:
    return obj
  else:
    if isinstance(obj, Message):
      # First, see if this object is a protocol buffer, which we can simply
      # serialize with the SerializeToString() call.
      return obj.SerializeToString()
    elif hasattr(obj, 'Proto'):
      return obj.Proto().SerializeToString()

def ResetWorkspace(root_folder=None):
  if root_folder is None:
    return cc_ResetWorkspace()
  else:
    if not os.path.exists(root_folder):
      os.makedirs(root_folder)
    return cc_ResetWorkspace(root_folder)

def CreateNet(net, input_blobs=[]):
  for input_blob in input_blobs:
    CreateBlob(input_blob)
  return cc_CreateNet(StringfyProto(net))

def RunOperatorOnce(operator):
  return cc_RunOperatorOnce(StringfyProto(operator))

def RunOperatorsOnce(operators):
  for op in operators:
    success = RunOperatorOnce(op)
    if not success:
      return False
  return True

def RunNetOnce(net):
  return cc_RunNetOnce(StringfyProto(net))

def RunPlan(plan):
  return cc_RunPlan(StringfyProto(plan))

def FeedBlob(name, arr, device_option=None):
  """Feeds a blob into the workspace.

  Inputs:
    name: the name of the blob.
    arr: either a TensorProto object or a numpy array object to be fed into the
        workspace.
    device_option (optional): the device option to feed the data with.
  Returns:
    True or False, stating whether the feed is successful.
  """
  if type(arr) is caffe2_pb2.TensorProto:
    arr = utils.Caffe2TensorToNumpyArray(arr)
  if device_option is not None:
    return cc_FeedBlob(name, arr, StringfyProto(device_option))
  else:
    return cc_FeedBlob(name, arr)


class Model(object):
  def __init__(self, net, parameters, inputs, outputs, device_option=None):
    """Initializes a model.

    Inputs:
      net: a Caffe2 NetDef protocol buffer.
      parameters: a TensorProtos object containing the parameters to feed into
          the network.
      inputs: a list of strings specifying the input blob names.
      outputs: a list of strings specifying the output blob names.
      device_option (optional): the device option used to run the model. If
          not given, we will use the net's device option.
    """
    self._name = net.name
    self._inputs = inputs
    self._outputs = outputs
    if device_option:
      self._device_option = device_option.SerializeToString()
    else:
      self._device_option = net.device_option.SerializeToString()
    # For a caffe2 net, before we create it, it needs to have all the parameter
    # blobs ready. The construction is in two steps: feed in all the parameters
    # first, and then create the network object.
    for param in parameters.protos:
      #print 'Feeding parameter', param.name
      FeedBlob(param.name, param, net.device_option)
    if not CreateNet(net, inputs):
      raise RuntimeError("Error when creating the model.")

  def Run(self, input_arrs):
    """Runs the model with the given input.

    Inputs:
      input_arrs: an iterable of input arrays.
    Outputs:
      output_arrs: a list of output arrays.
    """
    if len(input_arrs) != len(self._inputs):
      raise RuntimeError("Incorrect number of inputs.")
    for i, input_arr in enumerate(input_arrs):
      FeedBlob(self._inputs[i], input_arr, self._device_option)
    if not RunNet(self._name):
      raise RuntimeError("Error in running the network.")
    return [FetchBlob(s) for s in self._outputs]

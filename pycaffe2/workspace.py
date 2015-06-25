import atexit
from multiprocessing import Process
import socket

from .libcaffe2_python import *
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
  """Start a mint instance."""
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
    try:
      # First, see if this object is a protocol buffer, which we can simply
      # serialize with the SerializeToString() call.
      return obj.SerializeToString()
    except AttributeError:
      # Secind, see if this is an object defined in Pycaffe2, which exposes a
      # Proto() function that gives you the protocol buffer.
      return obj.Proto().SerializeToString()

def CreateNet(net):
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
  if device_option is not None:
    return cc_FeedBlob(name, arr, StringfyProto(device_option))
  else:
    return cc_FeedBlob(name, arr)
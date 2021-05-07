## @package muji
# Module caffe2.python.muji
"""muji.py does multi-gpu training for caffe2 with no need to change the c++
side code. Everything is defined on the computation graph level.

We support the following use cases:
  - 2 gpus, where peer access is enabled between them.
  - 4 gpus, where peer access are enabled between all of them.
  - 4 gpus, where peer access are enabled in two groups,
    between {1, 2} and {3, 4}
  - 8 gpus, where peer access are enabled in two groups,
    between {1, 2, 3, 4} and {5, 6, 7, 8}.
If above cases are not satisfied, a fallback function which does not rely on
peer access will be called.
"""

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace


def OnGPU(gpu_id):
    """A utility function that returns a device option protobuf of the
  specified gpu id.
  """
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = workspace.GpuDeviceType
    device_option.device_id = gpu_id
    return device_option


def OnCPU():
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CPU
    return device_option


def Allreduce(net, blobs, reduced_affix="_reduced", gpu_indices=None):
    """The general Allreduce interface that reroutes the function calls.
    CPUs and AMD GPUs are not supported because
    GetGpuPeerAccessPattern is called to get gpu peer access pattern.
  """
    if gpu_indices is None:
        gpu_indices = list(range(len(blobs)))
    if len(gpu_indices) != len(blobs):
        raise RuntimeError(
            "gpu_indices length and blobs length mismatch: %d vs %d" %
            (len(gpu_indices), len(blobs))
        )
    pattern = workspace.GetGpuPeerAccessPattern()
    if len(blobs) == 2 and pattern.shape[0] >= 2 and np.all(pattern[:2, :2]):
        return Allreduce2(net, blobs, reduced_affix, gpu_indices)
    elif len(blobs) == 4 and pattern.shape[0] >= 4 and np.all(pattern[:4, :4]):
        return Allreduce4(net, blobs, reduced_affix, gpu_indices)
    elif len(blobs) == 4 and pattern.shape[0] >= 4 and np.all(pattern[:2, :2]) and np.all(pattern[2:4, 2:4]):
        return Allreduce4Group2(net, blobs, reduced_affix, gpu_indices)
    elif len(blobs) == 8 and pattern.shape[0] >= 8 and np.all(pattern[:8, :8]):
        return Allreduce8(net, blobs, reduced_affix, gpu_indices)
    else:
        return AllreduceFallback(net, blobs, reduced_affix, gpu_indices)


def Allreduce2(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 2 gpus.

  Algorithm: 0r <- 0 + 1, 1r <- 0r, where r means "reduced"
  """
    a, b = blobs
    gpu_a, gpu_b = gpu_indices
    a_reduced = net.Add([a, b], a + reduced_affix, device_option=OnGPU(gpu_a))
    b_reduced = a_reduced.Copy(
        [],
        b + reduced_affix,
        device_option=OnGPU(gpu_b)
    )
    return a_reduced, b_reduced


def Allreduce4(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 4 gpus.

  Algorithm: 2 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3
      0r <- 0r + 2r
      2r <- 0r,
      1r <- 0r, 3r <- 2r
  """
    a, b, c, d = blobs
    gpu_a, gpu_b, gpu_c, gpu_d = gpu_indices
    # a_reduced <- a+b, c_reduced <- c + d
    a_reduced = net.Add(
        [a, b],
        str(a) + reduced_affix,
        device_option=OnGPU(gpu_a)
    )
    c_reduced = net.Add(
        [c, d],
        str(c) + reduced_affix,
        device_option=OnGPU(gpu_c)
    )
    # a_reduced <- a_reduced + c_reduced
    a_reduced = a_reduced.Add(c_reduced, a_reduced, device_option=OnGPU(gpu_a))
    # broadcast a_reduced to c_reduced
    c_reduced = a_reduced.Copy([], c_reduced, device_option=OnGPU(gpu_c))
    # broadcast to b and d
    b_reduced = a_reduced.Copy(
        [],
        str(b) + reduced_affix,
        device_option=OnGPU(gpu_b)
    )
    d_reduced = c_reduced.Copy(
        [],
        str(d) + reduced_affix,
        device_option=OnGPU(gpu_d)
    )
    return a_reduced, b_reduced, c_reduced, d_reduced


def Allreduce4Group2(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 4 gpus where peer access are enabled in {0,1} and {2,3}

  Algorithm: 2 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3
      0r <- 0r + 2r
      2r <- 0r,
      1r <- 0r, 3r <- 2r
  """
    a, b, c, d = blobs
    gpu_a, gpu_b, gpu_c, gpu_d = gpu_indices
    # a_reduced <- a+b, c_reduced <- c + d
    a_reduced = net.Add(
        [a, b],
        str(a) + reduced_affix,
        device_option=OnGPU(gpu_a)
    )
    c_reduced = net.Add(
        [c, d],
        str(c) + reduced_affix,
        device_option=OnGPU(gpu_c)
    )
    # copy from c_reduce(gpu_c) to c_reduce_copy(gpu_a)
    c_reduced_copy = c_reduced.Copy(
        [],
        str(c_reduced) + '_copy',
        device_option=OnGPU(gpu_a)
    )
    # a_reduced <- a_reduced + c_reduced_copy
    a_reduced = a_reduced.Add(c_reduced_copy, a_reduced, device_option=OnGPU(gpu_a))
    # broadcast a_reduced to c_reduced
    c_reduced = a_reduced.Copy([], c_reduced, device_option=OnGPU(gpu_c))
    # broadcast to b and d
    b_reduced = a_reduced.Copy(
        [],
        str(b) + reduced_affix,
        device_option=OnGPU(gpu_b)
    )
    d_reduced = c_reduced.Copy(
        [],
        str(d) + reduced_affix,
        device_option=OnGPU(gpu_d)
    )
    return a_reduced, b_reduced, c_reduced, d_reduced


def Allreduce8(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 8 gpus.

  Algorithm: 3 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3, 4r <- 4 + 5, 6r <- 6 + 7
      0r <- 0r + 2r, 4r <- 4r + 6r
      0r <- 0r + 4r
      4r <- 0r
      2r <- 0r, 6r <- 4r
      1r <- 0r, 3r <- 2r, 5r <- 4r, 7r <- 6r
  """
    reduced = [None] * 8
    # Reduction level 1
    for i in [0, 2, 4, 6]:
        reduced[i] = net.Add(
            [blobs[i], blobs[i + 1]],
            blobs[i] + reduced_affix,
            device_option=OnGPU(gpu_indices[i])
        )
    # Reduction level 2
    for i in [0, 4]:
        reduced[i] = net.Add(
            [reduced[i], reduced[i + 2]],
            str(blobs[i]) + reduced_affix,
            device_option=OnGPU(gpu_indices[i])
        )
    # Reduction level 3: this involves a copy.
    reduced_4_copy = reduced[4].Copy(
        [],
        str(reduced[4]) + '_copy',
        device_option=OnGPU(gpu_indices[0])
    )
    reduced[0] = reduced[0].Add(
        reduced_4_copy,
        reduced[0],
        device_option=OnGPU(gpu_indices[0])
    )
    # Broadcast level 1
    reduced[4] = reduced[0].Copy(
        [],
        reduced[4],
        device_option=OnGPU(gpu_indices[4])
    )
    # Broadcast level 2
    for i in [2, 6]:
        reduced[i] = reduced[i - 2].Copy(
            [],
            reduced[i],
            device_option=OnGPU(gpu_indices[i])
        )
    # Broadcast level 3
    for i in [1, 3, 5, 7]:
        reduced[i] = reduced[i - 1].Copy(
            [],
            blobs[i] + reduced_affix,
            device_option=OnGPU(gpu_indices[i])
        )
    return reduced


def AllreduceFallback(net, blobs, reduced_affix, gpu_indices):
    """A fallback option for Allreduce with no assumption on p2p.

  Algorithm: a flat operation on gpu 0
      0r <- 0
      0r <- 0r + i for i in gpu_indices[1:]
      ir <- 0r for i in gpu_indices[1:]
  """
    reduced = [None] * len(gpu_indices)
    if reduced_affix != '':
        # copy first
        reduced[0] = net.Copy(
            blobs[0],
            blobs[0] + reduced_affix,
            device_option=OnGPU(gpu_indices[0])
        )
    else:
        reduced[0] = blobs[0]
    # do temp copy and add
    temp_name = reduced[0] + '_temp_copy'
    for i in range(1, len(gpu_indices)):
        temp = net.Copy(
            blobs[i],
            temp_name,
            device_option=OnGPU(gpu_indices[0])
        )
        reduced[0] = net.Add(
            [temp, reduced[0]],
            reduced[0],
            device_option=OnGPU(gpu_indices[0])
        )
    # Broadcast to everyone else
    for i in range(1, len(gpu_indices)):
        reduced[i] = net.Copy(
            reduced[0],
            blobs[i] + reduced_affix,
            device_option=OnGPU(gpu_indices[i])
        )
    return reduced

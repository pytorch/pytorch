Tensor CUDA Stream API 
======================

A `CUDA Stream`_ is a linear sequence of execution that belongs to a specific CUDA device.
The PyTorch C++ API supports CUDA streams with the CUDAStream class and useful helper functions to make streaming operations easy.
You can find them in `CUDAStream.h`_. This note provides more details on how to use Pytorch C++ CUDA Stream APIs.

.. _CUDA Stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
.. _CUDAStream.h: https://pytorch.org/cppdocs/api/file_c10_cuda_CUDAStream.h.html#file-c10-cuda-cudastream-h
.. _CUDAStreamGuard.h: https://pytorch.org/cppdocs/api/structc10_1_1cuda_1_1_c_u_d_a_stream_guard.html

Acquiring CUDA stream
*********************

Pytorch C++ API provides the following ways to acquire a CUDA stream:

1. Acquire a new stream from the CUDA stream pool, streams are preallocated from the pool and returned in a round-robin fashion.

.. code-block:: cpp
  
  CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

.. tip::

  You can request a stream from the high priority pool by setting isHighPriority to true, or a stream for a specific device
  by setting device index (defaulting to the current CUDA stream's device index).

2. Acquire the default CUDA stream, for the passed CUDA device, or for the current device if no device index is passed.

.. code-block:: cpp
  
  CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

.. tip::
  
  The default stream is where most computation occurs when you aren't explicitly using streams.

3. Acquire the current CUDA stream, for the CUDA device with index ``device_index``, or for the current device if no device index is passed. 

.. code-block:: cpp
  
  CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

.. tip::
  
  The current CUDA stream will usually be the default CUDA stream for the device, but it may be different if someone
  called ``setCurrentCUDAStream`` or used ``StreamGuard`` or ``CUDAStreamGuard``.



Set CUDA stream
***************

Pytorch C++ API provides the following ways to set CUDA stream:

1. Set the current stream on the device of the passed in stream to be the passed in stream.

.. code-block:: cpp

  void setCurrentCUDAStream(CUDAStream stream);

.. attention::

  This function may have nosthing to do with the current device. It only changes the current stream on the stream's device.
  We recommend using ``CUDAStreamGuard``, instead, since it switches to the stream's device and makes it the current stream on that device.
  ``CUDAStreamGuard`` will also restore the current device and stream when it's destroyed

2. Use ``CUDAStreamGuard`` to switch to a CUDA stream within a scope, it is defined in `CUDAStreamGuard.h`_

.. tip::

  Use ``CUDAMultiStreamGuard`` if you need to set streams on multiple CUDA devices.

CUDA Stream Usage Examples
**************************

1. Acquiring and setting a CUDA stream

.. code-block:: cpp

  // acquire a new stream from CUDA stream pool on current device (device 0)
  // set current stream with this new stream and acquire current stream
  torch::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
  torch::cuda::setCurrentCUDAStream(myStream);
  torch::cuda::CUDAStream curStream = torch::cuda::getCurrentCUDAStream();
  assert(curStream == myStream);

  // acquire the default CUDA stream on current device (device 0)
  // set current stream with default stream and acquire current stream
  torch::cuda::CUDAStream defaultStream = torch::cuda::getDefaultCUDAStream();
  torch::cuda::setCurrentCUDAStream(defaultStream);
  curStream = torch::cuda::getCurrentCUDAStream();
  assert(curStream == defaultStream);
  assert(myStream != defaultStream)

.. attention::
  
  Above code is running on the same device/gpu (say device 0). `setCurrentCUDAStream` works as expected and set current CUDA stream correctly
  on device 0, this is because all the streams are on the same device. However, say above code is running on device 0 and we acquire `myStream`
  from device 1, then `torch::cuda::setCurrentCUDAStream(myStream)` will set `myStream` as current stream on device 1, not device 0.


2. Use various CUDA Guard to set CUDA stream

.. code-block:: cpp

   // create a vector of CUDA stream from current device (device 0)
   std::vector<torch::cuda::CUDAStream> streams0 = {torch::cuda::getDefaultCUDAStream(), torch::cuda::getStreamFromPool()};
   assert(streams0[0].device_index() == 0);
   assert(streams0[1].device_index() == 0);
   // set current stream as `streams0[0]` on `streams0[0]`'s current device (device 0)
   torch::cuda::setCurrentCUDAStream(streams0[0]);

   // create a vector of CUDA stream from device 1
   std::vector<torch::cuda::CUDAStream> streams1;
   {
     // create a CUDA Device guard within the scope to guard device 1
     torch::cuda::CUDAGuard device_guard(1);
     // acquire default CUDA stream from device 1
     streams1.push_back(torch::cuda::getDefaultCUDAStream());
     // acquire a new stream from CUDA stream pool on device 1
     streams1.push_back(torch::cuda::getStreamFromPool());
   }
   assert(streams1[0].device_index() == 1);
   assert(streams1[1].device_index() == 1);
   // set current stream as `streams1[0]` on `streams1[0]`'s current device (device 1)
   torch::cuda::setCurrentCUDAStream(streams1[0]);
   // current device is still 0 out of the scope of previous device guard
   assert(torch::cuda::current_device() == 0);

   // use CUDAStreamGuard to set a stream, it changes the stream on curernt devide and also passed in stream's device
   {
     torch::cuda::CUDAStreamGuard guard(streams1[1]);
     // current device is 1 within this scope
     assert(guard.current_device() == torch::Device(torch::kCUDA, 1));
     assert(torch::cuda::current_device() == 1);
     // current stream on device 1 is streams1[1], not streams1[0]
     assert(torch::cuda::getCurrentCUDAStream(1) == streams1[1]);
   }

   // device and stream are now back to the status before CUDAStreamGuard is created.
   assert(torch::cuda::current_device() == 0);
   assert(torch::cuda::getCurrentCUDAStream(1) == streams1[0]);

   // CUDA Device guard on changes the current device, but not the stream
   {
     // change current device to device 1
     torch::cuda::CUDAGuard guard(1);
     // current device beccomes to 1 within this scope
     assert(guard.current_device() == torch::Device(at::kCUDA, 1));
     assert(torch::cuda::current_device() == 1);
     // current stream on device 1 is still streams1[0] as what is set at the beginning, no change
     assert(torch::cuda::getCurrentCUDAStream(1) == streams1[0]);
   }

   // device go back to the status before above CUDAGuard is created
   assert(torch::cuda::current_device() == 0;

   // CUDAMultiStreamGuard can be used to set multiple streams on different device
   {
     // This is the same as calling `torch::cuda::setCurrentCUDAStream` on both streams
     torch::cuda::CUDAMultiStreamGuard({streams0[0], streams1[0]});
   }

   // CUDAMultiStreamGuard can be used to record original streams as well
   {
     torch::cuda::CUDAMultiStreamGuard guard;
     // assume we have 2 devices
     assert(guard.original_streams().size() == torch::cuda::getNumGPUs());
     // all the streams before the guard is created are recorded
     assert(guard.original_streams()[0] == streams0[0]);
     assert(guard.original_streams()[1] == streams1[0]);
   }

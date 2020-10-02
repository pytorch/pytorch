Tensor CUDA Stream API 
======================

A `CUDA Stream`_ is a linear sequence of execution that belongs to a specific device.
Pytorch C++ API has defined a CUDAStream class and a few useful functions to support CUDA Stream operations.
You can find them in `CUDAStream.h`_. This note provides more details on how to use Pytorch C++ CUDA Stream APIs.

.. _CUDA Stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
.. _CUDAStream.h: https://pytorch.org/cppdocs/api/file_c10_cuda_CUDAStream.h.html#file-c10-cuda-cudastream-h
.. _CUDAStreamGuard.h: https://pytorch.org/cppdocs/api/structc10_1_1cuda_1_1_c_u_d_a_stream_guard.html

Get CUDA stream
*****************

Pytorch C++ API provides the following ways to get CUDA stream:

1. Get a new stream from the CUDA stream pool, streams are preallocated from the pool and returned in a round-robin fashion.

.. code-block:: cpp
  
  CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

.. tip::

  You can request a stream from the high priority pool by setting isHighPriority to true, or a stream for a specific device
  by setting device (defaulting to the current CUDA stream).

2. Get the default CUDA stream, for the passed CUDA device, or for the current device if no device index is passed.

.. code-block:: cpp
  
  CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

.. tip::
  
  The default stream is where most computation occurs when you aren't explicitly using streams.

3. Get the current CUDA stream, for the passed CUDA device, or for the current device if no device index is passed. 

.. code-block:: cpp
  
  CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

.. tip::
  
  The current CUDA stream will usually be the default CUDA stream for the device, but it may be different if someone
  called ``setCurrentCUDAStream`` or used ``StreamGuard`` or ``CUDAStreamGuard``.



Set CUDA stream
*****************

Pytorch C++ API provides the following ways to set CUDA stream:

1. Set the current stream on the device of the passed in stream to be the passed in stream.

.. code-block:: cpp

  void setCurrentCUDAStream(CUDAStream stream);

.. tip::

  This function may have **nothing** to do with the current device, it toggles the current stream of the device of the passed stream.
  This depends on whether current device is the same as the current stream of the device of the passed stream.

.. attention::

  We prefer people to use ``CUDAStreamGuard`` (which will switch both your current device and current stream in the way you expect, and
  reset it back to its original state afterwards) instead of ``setCurrentCUDAStream``.

2. Use ``CUDAStreamGuard`` to set CUDA stream within a scope, it is defined in `CUDAStreamGuard.h`_

.. tip::
   If you need to set streams on multiple devices on CUDA, use ``CUDAMultiStreamGuard`` instead.


CUDA Stream Usage Examples
**************************

1. Get and Set CUDA stream

.. code-block:: cpp

  // get a new stream from CUDA stream pool on current device (assume current device is device 0)
  torch::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
  // set myStream as the current stream on myStream's current device (device 0 as well)
  // note that myStream's current device is the same as current device (within scope) here
  torch::cuda::setCurrentCUDAStream(myStream);
  // get current stream on current device (device 0)
  torch::cuda::CUDAStream curStream = torch::cuda::getCurrentCUDAStream();
  // now we have curStream == myStream on current device (device 0)
  assert(curStream == myStream);

  // get the default CUDA stream on current device (device 0)
  torch::cuda::CUDAStream defaultStream = torch::cuda::getDefaultCUDAStream();
  // set defaultStream as the current stream on defaultStream's current device (device 0 as well)
  // note that defaultStream's current device is the same as current device (within scope) here
  torch::cuda::setCurrentCUDAStream(defaultStream);
  // get current stream on current device (device 0) again
  curStream = torch::cuda::getCurrentCUDAStream();
  // now we have curStream == defaultStream
  assert(curStream == defaultStream);
  // myStream is different from  defaultStream since it get from stream pool that separate from default stream
  assert(myStream != defaultStream)

.. attention::
  
  Above code is running on the same device/gpu (say device 0). `setCurrentCUDAStream` works as expected and set current CUDA stream correctly
  on device 0, this is because all the streams are on the same device. However, say above code is running on device 0 and we get `myStream`
  from device 1, then `torch::cuda::setCurrentCUDAStream(myStream)` will set `myStream` as current stream on device 1, not device 0.


2. Use various CUDA Guard to set CUDA stream

.. code-block:: cpp

   // create a vector of CUDA stream from current device (assume this device 0)
   std::vector<torch::cuda::CUDAStream> streams0 = {torch::cuda::getDefaultCUDAStream(), torch::cuda::getStreamFromPool()};
   // both stream within the `streams0` are on device 0
   assert(streams0[0].device_index() == 0);
   assert(streams0[1].device_index() == 0);
   // set current stream as `streams0[0]` on `streams0[0]`'s current device (device 0)
   torch::cuda::setCurrentCUDAStream(streams0[0]);

   // use a CUDA Device guard to get CUDA streams from another device (device 1)
   std::vector<torch::cuda::CUDAStream> streams1;
   {
     // create a CUDA Device guard within the scope to guard device 1
     torch::cuda::CUDAGuard device_guard(1);
     // get default CUDA stream from device 1
     streams1.push_back(torch::cuda::getDefaultCUDAStream());
     // get a new stream from CUDA stream pool on device 1
     streams1.push_back(torch::cuda::getStreamFromPool());
   }
   // both stream within the `streams1` are on device 1
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
   // current device is back to 0
   assert(torch::cuda::current_device() == 0);
   // current stream on device 1 back to streams1[0]
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
   // current device is back to 0
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
     // all the streams be fore the guard is created are recorded
     assert(guard.original_streams()[0] == streams0[0]);
     assert(guard.original_streams()[1] == streams1[0]);
   }

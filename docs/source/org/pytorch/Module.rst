.. java:import:: com.facebook.jni HybridData

org.pytorch.Module (Module)
=============================

Source Code
------------

Full code can be found in `Github <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/Module.java>`_.

Overview
---------

Module is a wrapper of torch.jit.ScriptModule (`torch::jit::script::Module` in PyTorch C++ API)
which can be constructed with factory method load providing absolute path to the file with serialized TorchScript.

::

    IValue IValue.runMethod(String methodName, IValue... inputs)

for running a particular method of the script module.

::

    IValue IValue.forward(IValue... inputs)

Shortcut to run 'forward' method.

::

    IValue IValue.destroy()

Explicitly destructs native (C++) part of the Module, `torch::jit::script::Module`.
As fbjni library destructs native part automatically when current `org.pytorch.Module` instance will be collected by Java GC, the instance will not leak if this method is not called, but timing of deletion and the thread will be at the whim of the Java GC. If you want to control the thread and timing of the destructor, you should call this method explicitly.


Module API Details
------------------

.. java:package:: org.pytorch
   :noindex:

.. java:type:: public class Module

   Java holder for torch::jit::script::Module which owns it on jni side.

Methods
^^^^^^^
destroy
~~~~~~~~

.. java:method:: public void destroy()
   :outertype: Module

   Explicitly destructs native part. Current instance can not be used after this call. This method may be called multiple times safely. As fbjni library destructs native part automatically when current instance will be collected by Java GC, the instance will not leak if this method is not called, but timing of deletion and the thread will be at the whim of the Java GC. If you want to control the thread and timing of the destructor, you should call this method explicitly. \ :java:ref:`com.facebook.jni.HybridData.resetNative`\

forward
~~~~~~~~

.. java:method:: public IValue forward(IValue... inputs)
   :outertype: Module

   Runs 'forward' method of loaded torchscript module with specified arguments.

   :param inputs: arguments for torchscript module 'forward' method.
   :return: result of torchscript module 'forward' method evaluation

load
~~~~~~~~

.. java:method:: public static Module load(String modelAbsolutePath)
   :outertype: Module

   Loads serialized torchscript module from the specified absolute path on the disk.

   :param modelAbsolutePath: absolute path to file that contains the serialized torchscript module.
   :return: new \ :java:ref:`org.pytorch.Module`\  object which owns torch::jit::script::Module on jni side.

runMethod
~~~~~~~~

.. java:method:: public IValue runMethod(String methodName, IValue... inputs)
   :outertype: Module

   Runs specified method of loaded torchscript module with specified arguments.

   :param methodName: torchscript module method to run
   :param inputs: arguments that will be specified to torchscript module method call
   :return: result of torchscript module specified method evaluation

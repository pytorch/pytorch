.. java:import:: com.facebook.jni HybridData

Module
======

.. java:package:: org.pytorch
   :noindex:

.. java:type:: public class Module

   Java wrapper for torch::jit::Module.

Methods
-------
destroy
^^^^^^^

.. java:method:: public void destroy()
   :outertype: Module

   Explicitly destroys the native torch::jit::Module. Calling this method is not required, as the native object will be destroyed when this object is garbage-collected. However, the timing of garbage collection is not guaranteed, so proactively calling \ ``destroy``\  can free memory more quickly. See \ :java:ref:`com.facebook.jni.HybridData.resetNative`\ .

forward
^^^^^^^

.. java:method:: public IValue forward(IValue... inputs)
   :outertype: Module

   Runs the 'forward' method of this module with the specified arguments.

   :param inputs: arguments for the TorchScript module's 'forward' method.
   :return: return value from the 'forward' method.

load
^^^^

.. java:method:: public static Module load(String modelPath)
   :outertype: Module

   Loads a serialized TorchScript module from the specified path on the disk.

   :param modelPath: path to file that contains the serialized TorchScript module.
   :return: new \ :java:ref:`org.pytorch.Module`\  object which owns torch::jit::Module.

runMethod
^^^^^^^^^

.. java:method:: public IValue runMethod(String methodName, IValue... inputs)
   :outertype: Module

   Runs the specified method of this module with the specified arguments.

   :param methodName: name of the TorchScript method to run.
   :param inputs: arguments that will be passed to TorchScript method.
   :return: return value from the method.

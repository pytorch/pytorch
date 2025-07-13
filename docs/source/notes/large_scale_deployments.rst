Features for large-scale deployments
====================================

.. contents:: :local:

This note talks about several extension points and tricks that might be useful
when running PyTorch within a larger system or operating multiple systems using
PyTorch in a larger organization.

The note assumes that you either build PyTorch from source in your
organization or have an ability to statically link additional code to be loaded
when PyTorch is used. Therefore, many of the hooks are exposed as C++ APIs that
can be triggered once in a centralized place, e.g. in static initialization
code.

Fleet-wide operator profiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch comes with :mod:`torch.autograd.profiler` capable of measuring time
taken by individual operators on demand. One can use the same mechanism to do
"always ON" measurements for any process running PyTorch. It might be useful for
gathering information about PyTorch workloads running in a given process or
across the entire set of machines.

New callbacks for any operator invocation can be added with
``torch::addGlobalCallback``. Hooks will be called with
``torch::RecordFunction`` struct that describes invocation
context (e.g. `name`). If enabled, ``RecordFunction::inputs()`` contains arguments
of the function represented as ``torch::IValue`` variant type. Note, that inputs
logging is relatively expensive and thus has to be enabled explicitly.

The operator callbacks also have access to ``c10::ThreadLocalDebugInfo::get()``
interface that returns a pointer to the struct holding the debug information.
This debug information can be set earlier by using ``at::DebugInfoGuard`` object.
Debug information is propagated through the forward (including async ``fork``
tasks) and backward passes and can be useful for passing some extra information
about execution environment (e.g. model id) from the higher layers of the
application down to the operator callbacks.

Invoking callbacks adds some overhead, so usually it's useful to just randomly
sample operator invocations. This can be enabled on per-callback basis with an
optional sampling rate passed into ``torch::addGlobalCallback``.

Note, that ``addGlobalCallback`` is not thread-safe and can be called only when no
PyTorch operator is running. Usually, it's a good idea to call them once during
initialization.

Here's an example:

.. code-block:: cpp

    // Called somewhere in the program beginning
    void init() {
        // Sample one in a hundred operator runs randomly
        addGlobalCallback(
          RecordFunctionCallback(
            &onFunctionEnter,
            &onFunctionExit)
          .needsInputs(true)
          .samplingProb(0.01)
        );
        // Note, to enable observers in the model calling thread,
        // call enableRecordFunction() in the thread before running a model
    }

    void onFunctionEnter(const RecordFunction& fn) {
        std::cerr << "Before function " << fn.name()
                  << " with " << fn.inputs().size() << " inputs" << std::endl;
    }

    void onFunctionExit(const RecordFunction& fn) {
        std::cerr << "After function " << fn.name();
    }

API usage logging
^^^^^^^^^^^^^^^^^

When running in a broader ecosystem, for example in managed job scheduler, it's
often useful to track which binaries invoke particular PyTorch APIs. There
exists simple instrumentation injected at several important API points that
triggers a given callback. Because usually PyTorch is invoked in one-off python
scripts, the callback fires only once for a given process for each of the APIs.

``c10::SetAPIUsageHandler`` can be used to register API usage instrumentation
handler. Passed argument is going to be an "api key" identifying used point, for
example ``python.import`` for PyTorch extension import.

.. code-block:: cpp

    SetAPIUsageLogger([](const std::string& event_name) {
        std::cerr << "API was used: " << event_name << std::endl;
    });

Note for developers: new API trigger points can be added in code with
``C10_LOG_API_USAGE_ONCE("my_api")`` in C++ or
``torch._C._log_api_usage_once("my.api")`` in Python.


Common extension points
^^^^^^^^^^^^^^^^^^^^^^^

PyTorch APIs are generally loosely coupled and it's easy to replace a component
with specialized version. Common extension points include:

* Custom operators implemented in C++ - see `tutorial for more details <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_.
* Custom data reading can be often integrated directly by invoking corresponding python library. Existing functionality of :mod:`torch.utils.data` can be utilized by extending :class:`~torch.utils.data.Dataset` or :class:`~torch.utils.data.IterableDataset`.

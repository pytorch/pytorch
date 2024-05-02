.. _multiprocessing-best-practices:

Multiprocessing best practices
==============================

:mod:`torch.multiprocessing` is a drop in replacement for Python's
:mod:`python:multiprocessing` module. It supports the exact same operations,
but extends it, so that all tensors sent through a
:class:`python:multiprocessing.Queue`, will have their data moved into shared
memory and will only send a handle to another process.

.. note::

    When a :class:`~torch.Tensor` is sent to another process, the
    :class:`~torch.Tensor` data is shared. If :attr:`torch.Tensor.grad` is
    not ``None``, it is also shared. After a :class:`~torch.Tensor` without
    a :attr:`torch.Tensor.grad` field is sent to the other process, it
    creates a standard process-specific ``.grad`` :class:`~torch.Tensor` that
    is not automatically shared across all processes, unlike how the
    :class:`~torch.Tensor`'s data has been shared.

This allows to implement various training methods, like Hogwild, A3C, or any
others that require asynchronous operation.

.. _multiprocessing-cuda-note:

CUDA in multiprocessing
-----------------------

The CUDA runtime does not support the ``fork`` start method; either the ``spawn`` or ``forkserver`` start method are
required to use CUDA in subprocesses.

.. note::
  The start method can be set via either creating a context with
  ``multiprocessing.get_context(...)`` or directly using
  ``multiprocessing.set_start_method(...)``.

Unlike CPU tensors, the sending process is required to keep the original tensor
as long as the receiving process retains a copy of the tensor. It is implemented
under the hood but requires users to follow the best practices for the program
to run correctly. For example, the sending process must stay alive as long as
the consumer process has references to the tensor, and the refcounting can not
save you if the consumer process exits abnormally via a fatal signal. See
:ref:`this section <multiprocessing-cuda-sharing-details>`.

See also: :ref:`cuda-nn-ddp-instead`


Best practices and tips
-----------------------

Avoiding and fighting deadlocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a lot of things that can go wrong when a new process is spawned, with
the most common cause of deadlocks being background threads. If there's any
thread that holds a lock or imports a module, and ``fork`` is called, it's very
likely that the subprocess will be in a corrupted state and will deadlock or
fail in a different way. Note that even if you don't, Python built in
libraries do - no need to look further than :mod:`python:multiprocessing`.
:class:`python:multiprocessing.Queue` is actually a very complex class, that
spawns multiple threads used to serialize, send and receive objects, and they
can cause aforementioned problems too. If you find yourself in such situation
try using a :class:`~python:multiprocessing.queues.SimpleQueue`, that doesn't
use any additional threads.

We're trying our best to make it easy for you and ensure these deadlocks don't
happen but some things are out of our control. If you have any issues you can't
cope with for a while, try reaching out on forums, and we'll see if it's an
issue we can fix.

Reuse buffers passed through a Queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remember that each time you put a :class:`~torch.Tensor` into a
:class:`python:multiprocessing.Queue`, it has to be moved into shared memory.
If it's already shared, it is a no-op, otherwise it will incur an additional
memory copy that can slow down the whole process. Even if you have a pool of
processes sending data to a single one, make it send the buffers back - this
is nearly free and will let you avoid a copy when sending next batch.

Asynchronous multiprocess training (e.g. Hogwild)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using :mod:`torch.multiprocessing`, it is possible to train a model
asynchronously, with parameters either shared all the time, or being
periodically synchronized. In the first case, we recommend sending over the whole
model object, while in the latter, we advise to only send the
:meth:`~torch.nn.Module.state_dict`.

We recommend using :class:`python:multiprocessing.Queue` for passing all kinds
of PyTorch objects between processes. It is possible to e.g. inherit the tensors
and storages already in shared memory, when using the ``fork`` start method,
however it is very bug prone and should be used with care, and only by advanced
users. Queues, even though they're sometimes a less elegant solution, will work
properly in all cases.

.. warning::

    You should be careful about having global statements, that are not guarded
    with an ``if __name__ == '__main__'``. If a different start method than
    ``fork`` is used, they will be executed in all subprocesses.

Hogwild
~~~~~~~

A concrete Hogwild implementation can be found in the `examples repository`__,
but to showcase the overall structure of the code, there's also a minimal
example below as well::

    import torch.multiprocessing as mp
    from model import MyModel

    def train(model):
        # Construct data_loader, optimizer, etc.
        for data, labels in data_loader:
            optimizer.zero_grad()
            loss_fn(model(data), labels).backward()
            optimizer.step()  # This will update the shared parameters

    if __name__ == '__main__':
        num_processes = 4
        model = MyModel()
        # NOTE: this is required for the ``fork`` method to work
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

.. __: https://github.com/pytorch/examples/tree/master/mnist_hogwild



CPU in multiprocessing
----------------------

Inappropriate multiprocessing can lead to CPU oversubscription, causing
different processes to compete for CPU resources, resulting in low
efficiency.

This tutorial will explain what CPU oversubscription is and how to
avoid it.

CPU oversubscription
^^^^^^^^^^^^^^^^^^^^

CPU oversubscription is a technical term that refers to a situation
where the total number of vCPUs allocated to a system exceeds the total
number of vCPUs available on the hardware.

This leads to severe contention for CPU resources. In such cases, there
is frequent switching between processes, which increases processes
switching overhead and decreases overall system efficiency.

See CPU oversubscription with the code examples in the Hogwild
implementation found in the `example
repository <https://github.com/pytorch/examples/tree/main/mnist_hogwild>`__.

When running the training example with the following command on CPU
using 4 processes:

.. code-block:: bash

   python main.py --num-processes 4

Assuming there are N vCPUs available on the machine, executing the above
command will generate 4 subprocesses. Each subprocess will allocate N
vCPUs for itself, resulting in a requirement of 4*N vCPUs. However, the
machine only has N vCPUs available. Consequently, the different
processes will compete for resources, leading to frequent process
switching.

The following observations indicate the presence of CPU over
subscription:

#. High CPU Utilization: By using the ``htop`` command, you can observe
   that the CPU utilization is consistently high, often reaching or
   exceeding its maximum capacity. This indicates that the demand for
   CPU resources exceeds the available physical cores, causing
   contention and competition among processes for CPU time.

#. Frequent Context Switching with Low System Efficiency: In an
   oversubscribed CPU scenario, processes compete for CPU time, and the
   operating system needs to rapidly switch between different processes
   to allocate resources fairly. This frequent context switching adds
   overhead and reduces the overall system efficiency.

Avoid CPU oversubscription
^^^^^^^^^^^^^^^^^^^^^^^^^^

A good way to avoid CPU oversubscription is proper resource allocation.
Ensure that the number of processes or threads running concurrently does
not exceed the available CPU resources.

In this case, a solution would be to specify the appropriate number of
threads in the subprocesses. This can be achieved by setting the number
of threads for each process using the ``torch.set_num_threads(int)``
function in subprocess.

Assuming there are N vCPUs on the machine and M processes will be
generated, the maximum ``num_threads`` value used by each process would
be ``floor(N/M)``. To avoid CPU oversubscription in the mnist_hogwild
example, the following changes are needed for the file ``train.py`` in
`example
repository <https://github.com/pytorch/examples/tree/main/mnist_hogwild>`__.

.. code:: python

   def train(rank, args, model, device, dataset, dataloader_kwargs):
       torch.manual_seed(args.seed + rank)

       #### define the num threads used in current sub-processes
       torch.set_num_threads(floor(N/M))

       train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

       optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
       for epoch in range(1, args.epochs + 1):
           train_epoch(epoch, args, model, device, train_loader, optimizer)

Set ``num_thread`` for each process using
``torch.set_num_threads(floor(N/M))``. where you replace N with the
number of vCPUs available and M with the chosen number of processes. The
appropriate ``num_thread`` value will vary depending on the specific
task at hand. However, as a general guideline, the maximum value for the
``num_thread`` should be ``floor(N/M)`` to avoid CPU oversubscription.
In the `mnist_hogwild <https://github.com/pytorch/examples/tree/main/mnist_hogwild>`__ training example, after avoiding CPU over
subscription, you can achieve a 30x performance boost.

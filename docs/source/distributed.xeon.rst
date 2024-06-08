torch.backends.xeon.run_cpu
===========================

There are a set of configurations that would influence the performance of PyTorch inference running on Intel(R) Xeon(R) Scalable Processors.
To get peak performance, the ``torch.backends.xeon.run_cpu`` script is provided that optimizes the configuration of thread and memory management.
For thread management, the script configures thread affinity and the preload of Intel(R) OMP library.
For memory management, it configures NUMA binding and preloads optimized memory allocation libraries (e.g. tcmalloc, jemalloc).
In addition, the script provides tunable parameters for compute resource allocation in both single instance and multiple instance scenarios,
helping the users try out an optimal coordination of resource utilization for the specific workloads.

Prerequisites
-------------

NUMA Access Control
~~~~~~~~~~~~~~~~~~~

It is a good thing that more and more CPU cores are provided to users in one socket, as it brings in more computation resources.
However, this also brings memory access competitions. Program can stall because memory is busy to visit.
To address this problem, Non-Uniform Memory Access (NUMA) was introduced.
Comparing to Uniform Memory Access (UMA), in which scenario all the memories are connected to all cores equally,
NUMA tells memories into multiple groups. Certain number of memories are directly attached to one socket's integrated memory controller to become local memory of this socket.
Local memory access is much faster than remote memory access.

Users can get CPU information with ``lscpu`` command on Linux to learn how many cores, sockets are there on the machine.
Also, NUMA information like how CPU cores are distributed can also be retrieved.
The following is an example of ``lscpu`` execution on a machine with Intel (R) Xeon (R) CPU Max 9480.
2 sockets were detected. Each socket has 56 physical cores onboard. Since Hyper-Threading is enabled, each core can run 2 threads.
i.e. each socket has another 56 logical cores. Thus, there are 224 CPU cores on service.
When indexing CPU cores, usually physical cores are indexed before logical core.
In this case, the first 56 cores (0-55) are physical cores on the first NUMA socket (node), the second 56 cores (56-111) are physical cores on the second NUMA socket (node).
Logical cores are indexed afterward. 112-167 are 56 logical cores on the first NUMA socket,
168-223 are the second 56 logical cores on the second NUMA socket.
Typically, running PyTorch programs with compute intense workloads should avoid using logical cores to get good performance.

.. code-block:: console

   $ lscpu
   ...
   CPU(s):                  224
     On-line CPU(s) list:   0-223
   Vendor ID:               GenuineIntel
     Model name:            Intel (R) Xeon (R) CPU Max 9480
       CPU family:          6
       Model:               143
       Thread(s) per core:  2
       Core(s) per socket:  56
       Socket(s):           2
   ...
   NUMA:
     NUMA node(s):          2
     NUMA node0 CPU(s):     0-55,112-167
     NUMA node1 CPU(s):     56-111,168-223
   ...

Linux provides a tool, ``numactl``, that allows user control of NUMA policy for processes or shared memory.
It runs processes with a specific NUMA scheduling or memory placement policy.
As described above, cores share high-speed cache in one socket, thus it is a good idea to avoid cross socket computations.
From a memory access perspective, bounding memory access locally is much faster than accessing remote memories.
``numactl`` command should have been installed in recent Linux distributions. In case it is missing, we can install it manually with the installation command, like

.. code-block:: console

   $ apt-get install numactl

on Ubuntu, or

.. code-block:: console

   $ yum install numactl

on CentOS.

The ``taskset`` command in Linux is another powerful utility that allows you to set or retrieve the CPU affinity of a running process.
``taskset`` are pre-installed in most Linux distributions and in case it's not, we can install it with command

.. code-block:: console

   $ apt-get install util-linux

on Ubuntu, or

.. code-block:: console

   $ yum install util-linux

on CentOS.

OpenMP
~~~~~~

OpenMP is an implementation of multithreading, a method of parallelizing where a primary thread (a series of instructions executed consecutively) forks a specified number of sub-threads and the system divides a task among them. The threads then run concurrently, with the runtime environment allocating threads to different processors.
Users can control OpenMP behaviors with some environment variable settings to fit for their workloads, the settings are read and executed by OMP libraries. By default, PyTorch uses GNU OpenMP Library (GNU libgomp) for parallel computation. On Intel(R) platforms, Intel(R) OpenMP Runtime Library (libiomp) provides OpenMP API specification support. It usually brings more performance benefits compared to libgomp.

The Intel(R) OpenMP Runtime Library can be installed via the command

.. code-block:: console

   $ pip install intel-openmp

or

.. code-block:: console

   $ conda install mkl

Memory Allocator
~~~~~~~~~~~~~~~~

Memory allocator plays an important role from performance perspective as well. A more efficient memory usage reduces overhead on unnecessary memory allocations or destructions, and thus results in a faster execution. From practical experiences, for deep learning workloads, JeMalloc or TCMalloc can get better performance by reusing memory as much as possible than default malloc function.

TCMalloc can be installed by

.. code-block:: console

   $ apt-get install google-perftools

on Ubuntu, or

.. code-block:: console

   $ yum install gperftools

on CentOS.

In conda environment, it can also be installed by

.. code-block:: console

   $ conda install conda-forge::gperftools

JeMalloc can be installed by

.. code-block:: console

   $ apt-get install libjemalloc2

on Ubuntu, or

.. code-block:: console

   $ yum install jemalloc

on CentOS, or

.. code-block:: console

   $ conda install conda-forge::jemalloc

in conda environment.


Quick Start Example Commands
----------------------------

1. To run single-instance inference with 1 thread on 1 CPU core (only Core #0 would be used)

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance 1 <program.py> [program_args]

2. To run single-instance inference on a single CPU node (NUMA socket).

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --node-id 0 <program.py> [program_args]

3. To run multi-instance inference, 8 instances with 14 cores per instance on a 112-core CPU

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --ninstances 8 --ncores-per-instance 14 <program.py> [program_args]

4. To run inference in throughput mode, in which all the cores in each CPU node set up an instance

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --throughput-mode <program.py> [program_args]

.. note::

   Term "instance" here doesn't refer to a cloud instance. This script is executed as a single process which invokes multiple "instances" which are formed from multiple threads. "Instance" is kind of group of threads in this context.

Usage of torch.backends.xeon.run_cpu
------------------------------------

The argument list and usage guidance can be shown with

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu –h
   usage: run_cpu.py [-h] [--multi-instance] [-m] [--no-python] [--enable-tcmalloc] [--enable-jemalloc] [--use-default-allocator] [--disable-iomp] [--ncores-per-instance] [--ninstances] [--skip-cross-node-cores] [--rank] [--latency-mode] [--throughput-mode] [--node-id] [--use-logical-core] [--disable-numactl] [--disable-taskset] [--core-list] [--log-path] [--log-file-prefix] <program> [program_args]

positional arguments
~~~~~~~~~~~~~~~~~~~~

+------------------+---------------------------------------------------------+
| knob             | help                                                    |
+==================+=========================================================+
| ``program``      | The full path of the program/script to be launched.     |
+------------------+---------------------------------------------------------+
| ``program_args`` | All the arguments for the program/script to be launched.|
+------------------+---------------------------------------------------------+

Explanation of the options
~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic option settings (knobs) are:

+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+
| knob                 | type | default value | help                                                                                                                    |
+======================+======+===============+=========================================================================================================================+
| ``-h``, ``--help``   |      |               | Show the help message and exit.                                                                                         |
+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+
| ``-m``, ``--module`` |      |               | Changes each process to interpret the launch script as a python module, executing with the same behavior as 'python -m'.|
+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--no-python``      | bool | False         | Do not prepend the program with "python" - just exec it directly. Useful when the script is not a Python script.        |
+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--log-path``       | str  | ''            | The log file directory. Default path is ``''``, which means disable logging to files.                                   |
+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--log-file-prefix``| str  | 'run'         | log file name prefix.                                                                                                   |
+----------------------+------+---------------+-------------------------------------------------------------------------------------------------------------------------+

Knobs for applying or disabling optimizations are:

+-----------------------------+------+---------------+--------------------------------------------------------------------------------------------------------------------------+
| knob                        | type | default value | help                                                                                                                     |
+=============================+======+===============+==========================================================================================================================+
| ``--enable-tcmalloc``       | bool | False         | Enable ``TCMalloc`` memory allocator.                                                                                    |
+-----------------------------+------+---------------+--------------------------------------------------------------------------------------------------------------------------+
| ``--enable-jemalloc``       | bool | False         | Enable ``JeMalloc`` memory allocator.                                                                                    |
+-----------------------------+------+---------------+--------------------------------------------------------------------------------------------------------------------------+
| ``--use-default-allocator`` | bool | False         | Use default memory allocator. Neither ``TCMalloc`` nor ``JeMalloc`` would be used.                                       |
+-----------------------------+------+---------------+--------------------------------------------------------------------------------------------------------------------------+
| ``--disable-iomp``          | bool | False         | By default, Intel(R) OpenMP lib will be used if installed. Setting this flag would disable the usage of Intel(R) OpenMP. |
+-----------------------------+------+---------------+--------------------------------------------------------------------------------------------------------------------------+

.. note::

   Memory allocator influences performance. If users do not specify a desired memory allocator, the ``run_cpu`` script will search if any of them is installed in the order of TCMalloc > JeMalloc > PyTorch default memory allocator, and takes the first matched one.

Knobs for controlling instance number and compute resource allocation are:

+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| knob                        | type | default value | help                                                                                                                                         |
+=============================+======+===============+==============================================================================================================================================+
| ``--ninstances``            | int  | 0             | Number of instances.                                                                                                                         |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--ncores-per-instance``   | int  | 0             | Number of cores used by every instance.                                                                                                      |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--node-id``               | int  | -1            | Node id for multi-instance, by default all nodes will be used.                                                                               |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--core-list``             | str  | ''            | Specify the core list as "core_id, core_id, ...." or core range as "core_id-core_id". By dafault all the cores will be used.                 |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--use-logical-core``      | bool | False         | By default only physical cores are used. Specify this flag to use logical cores.                                                             |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--skip-cross-node-cores`` | bool | False         | Prevent the workload to be executed on cores across NUMA nodes.                                                                              |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--rank``                  | int  | -1            | Specify instance index to assign ncores_per_instance for rank; otherwise ncores_per_instance will be assigned sequentially to the instances. |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--multi-instance``        | bool | False         | A quick set to invoke multiple instances of the workload on multi-socket CPU servers.                                                        |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--latency-mode``          | bool | False         | A quick set to invoke benchmarking with latency mode, in which all physical cores are used and 4 cores per instance.                         |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--throughput-mode``       | bool | False         | A quick set to invoke benchmarking with throughput mode, in which all physical cores are used and 1 numa node per instance.                  |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--disable-numactl``       | bool | False         | By default ``numactl`` command is used to control NUMA access. Setting this flag will disable it.                                            |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| ``--disable-taskset``       | bool | False         | Disable the usage of ``taskset`` command.                                                                                                    |
+-----------------------------+------+---------------+----------------------------------------------------------------------------------------------------------------------------------------------+

.. note::

   Environment variables that will be set by this script include

   +------------------+-------------------------------------------------------------------------------------------------+
   | Environ Variable |                                             Value                                               |
   +==================+=================================================================================================+
   |    LD_PRELOAD    | Depending on knobs you set, <lib>/libiomp5.so, <lib>/libjemalloc.so, <lib>/libtcmalloc.so might |
   |                  | be appended to LD_PRELOAD.                                                                      |
   +------------------+-------------------------------------------------------------------------------------------------+
   |   KMP_AFFINITY   | If libiomp5.so is preloaded, KMP_AFFINITY could be set to "granularity=fine,compact,1,0".       |
   +------------------+-------------------------------------------------------------------------------------------------+
   |   KMP_BLOCKTIME  | If libiomp5.so is preloaded, KMP_BLOCKTIME is set to "1".                                       |
   +------------------+-------------------------------------------------------------------------------------------------+
   |  OMP_NUM_THREADS | value of ncores_per_instance                                                                    |
   +------------------+-------------------------------------------------------------------------------------------------+
   |    MALLOC_CONF   | If libjemalloc.so is preloaded, MALLOC_CONF will be set to                                      |
   |                  | "oversize_threshold:1,background_thread:true,metadata_thp:auto".                                |
   +------------------+-------------------------------------------------------------------------------------------------+

   Please note that the script respects environment variables set preliminarily. i.e. If you have set the environment variables mentioned above before running the script, the values of the variables will not overwritten by the script.

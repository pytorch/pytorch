Quickstart
===========

To launch a **fault-tolerant** job, run the following on all nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=NUM_NODES
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=c10d
            --rdzv_endpoint=HOST_NODE_ADDR
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


To launch an **elastic** job, run the following on at least ``MIN_SIZE`` nodes
and at most ``MAX_SIZE`` nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=MIN_SIZE:MAX_SIZE
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=c10d
            --rdzv_endpoint=HOST_NODE_ADDR
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400),
specifies the node and the port on which the C10d rendezvous backend should be
instantiated and hosted. It can be any node in your training cluster, but
ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

.. note::
   The ``--standalone`` option can be passed to launch a single node job with a
   sidecar rendezvous backend. You don’t have to pass ``--rdzv_id``,
   ``--rdzv_endpoint``, and ``--rdzv_backend`` when the ``--standalone`` option
   is used.


.. note::
   Learn more about writing your distributed training script
   `here <train_script.html>`_.

If ``torch.distributed.run`` does not meet your requirements you may use our
APIs directly for more powerful customization. Start by taking a look at the
`elastic agent <agent.html>`_ API).

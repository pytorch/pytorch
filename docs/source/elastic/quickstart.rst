Quickstart
===========

.. code-block:: bash

   pip install torch

   # start a single-node etcd server on ONE host
   etcd --enable-v2
        --listen-client-urls http://0.0.0.0:2379,http://127.0.0.1:4001
        --advertise-client-urls PUBLIC_HOSTNAME:2379

To launch a **fault-tolerant** job, run the following on all nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=NUM_NODES
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


To launch an **elastic** job, run the following on at least ``MIN_SIZE`` nodes
and at most ``MAX_SIZE`` nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=MIN_SIZE:MAX_SIZE
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


.. note:: The `--standalone` option can be passed to launch a single node job with
          a sidecar rendezvous server. You don’t have to pass —rdzv_id, —rdzv_endpoint,
          and —rdzv_backend when the —standalone option is used


.. note:: Learn more about writing your distributed training script
          `here <train_script.html>`_.

If ``torch.distributed.run`` does not meet your requirements
you may use our APIs directly for more powerful customization. Start by
taking a look at the `elastic agent <agent.html>`_ API).

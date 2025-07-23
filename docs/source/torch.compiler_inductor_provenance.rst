.. _torchinductor-provenance:

TorchInductor and AOTInductor Provenance Tracking
=================================================

.. warning::
    This feature is a prototype under active development and there will be
    breaking change in future releases.
    The current compatibility of this tool is limited to the latest nightly build of PyTorch.


This section describes how to use the provenance tracking feature for TorchInductor and AOTInductor in ``tlparse``.
Provenance tracking helps you visualize the relationships between the input GraphModule to (AOT)Inductor and the optimized code generated. This feature allows you to trace how your original operations are transformed during compilation.

Some example screenshots of the provenance tracking tool are shown below.
The tool visualizes the mapping between nodes in the input graph (panel 1), the post grad graph (panel 2), and the Inductor generated code (panel 3).

The **bolded** lines represent nodes/kernels covered by the current provenance tracing functionality.
We currently cover triton kernels, cpp kernels, and combo kernels.
The yellow highlighting shows the provenance of the nodes/kernels.


Example screenshot of the provenance tracking tool for TorchInductor:
 .. image:: _static/img/inductor_provenance/provenance_jit_inductor.png

Example screenshot of the provenance tracking tool for AOTInductor:
 .. image:: _static/img/inductor_provenance/provenance_aot_inductor.png


Using the Provenance Tracking Highlighter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these steps to enable and use provenance tracking in your PyTorch project:

1. Install ``tlparse`` by ``cargo install tlparse``. If you don't have ``cargo``, see `The Cargo Book <https://doc.rust-lang.org/cargo/getting-started/installation.html>`__ for instructions to install.
2. Run your program with required flags:

   .. code-block:: bash

     TORCH_TRACE=~/my_trace_log_dir TORCH_LOGS="+inductor" TORCH_COMPILE_DEBUG=1 python your_program.py

   This will generate a log file in ``~/my_trace_log_dir``. The log file will be used by tlparse to generate the provenance tracking highlighter.
3. Run ``tlparse`` on the log with ``--inductor-provenance`` flag. For example:

   .. code-block:: bash

      tlparse log_file_name.log --inductor-provenance

   - Even if you don't add the ``--inductor-provenance`` flag, you should be able to see the mapping in json format in the ``inductor_provenance_tracking_node_mappings_<number>.json`` file in the ``index.html`` tlparse output.
   - Run ``tlpare`` directly on the log file. It might not work if you run "tlparse parse <folder_name>  --inductor-provenance".
   - The ``tlparse`` artifacts used by the provenance tracking highlighter are:

      * ``before_pre_grad_graph.txt``
      * ``after_post_grad_graph.txt``
      * ``inductor_aot_wrapper_code.txt``
      * ``inductor_output_code.txt``
      * ``inductor_provenance_tracking_node_mappings.json``

After running ``tlparse <file_name> --inductor-provenance``, you should see an additional "Provenance Tracking" section in the tlparse output. Clicking into the link(s) to access the provenance tracking tool.
For a demo, see: https://github.com/pytorch/tlparse/pull/93

 .. image:: _static/img/inductor_provenance/index.png


See Also
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tlparse`` is a tool written in Rust.

- Link to the tlparse GitHub repo: https://github.com/pytorch/tlparse
- Learn more about ``tlparse`` at :ref:`torch.compiler_troubleshooting`

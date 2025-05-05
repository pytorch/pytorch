.. _torchinductor-provenance:

TorchInductor and AOTInductor Provenance Tracking
=================================================

This section describes how to use the provenance tracking feature for TorchInductor and AOTInductor in ``tlparse``.
Some example screenshots of the provenance tracking tool are shown below.
The tool visualizes the mapping between nodes in the input graph (panel 1), the post grad graph (panel 2), and the Inductor generated code (panel 3).

The **bolded** lines represent nodes/kernels covered by the current provenance tracing functionality.
We currently cover triton kernels, cpp kernels, and combo kernels.
The yellow highlighting shows the provenance of the nodes/kernels.


Example screenshot of the provenance tracking tool for TorchInductor:
 .. image:: _static/img/inductor_provenance/provenance_jit_inductor.png

Example screenshot of the provenance tracking tool for AOTInductor:
 .. image:: _static/img/inductor_provenance/provenance_aot_inductor.png


Get Started
~~~~~~~~~~~



-  Use the following flags when running your program to produce necessary artifacts: ``TORCH_TRACE=~/my_trace_log_dir  TORCH_LOGS="+inductor"  TORCH_COMPILE_DEBUG=1``

   -  These flags will produce a log file in ``~/my_trace_log_dir``. The log file will be used by tlparse to generate the provenance tracking highlighter.


- Then run ``tlparse`` on the log with ``--inductor-provenance`` flag. For example, ``tlparse log_file_name.log --inductor-provenance``.

   - See a demo video at https://github.com/pytorch/tlparse/pull/93.
   - Even if you don't add the --inductor-provenance flag, you should be able to see the mapping in json format in the ``inductor_provenance_tracking_node_mappings_<number>.json`` file in the ``index.html`` tlparse output.
   - Please run ``tlpare`` directly on the log file. It might not work if you run "tlparse parse <folder_name>  --inductor-provenance".
   - The ``tlparse`` artifacts used by the provenance tracking highlighter are: inductor_pre_grad_graph.txt, inductor_post_grad_graph.txt, inductor_aot_wrapper_code.txt, inductor_output_code,txt inductor_provenance_tracking_node_mappings.json.


After running ``tlparse <file_name> --inductor-provenance``, you should see an additional "Provenance Tracking" section in the tlparse output. Clicking into the link(s) to access the provenance tracking tool.

 .. image:: _static/img/inductor_provenance/index.png


More about tlparse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tlparse`` is a tool written in Rust, and can be installed by ``cargo install tlparse``.

- Link to the tlparse GitHub repo: https://github.com/pytorch/tlparse
- Learn more about ``tlparse`` at :ref:`torch.compiler_troubleshooting`

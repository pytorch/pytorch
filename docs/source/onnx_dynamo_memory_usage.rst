Understanding TorchDynamo-based ONNX Exporter Memory Usage
==========================================================
The TorchDynamo-based ONNX exporter utilizes torch.export.export() function to leverage
`FakeTensorMode <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_ to avoid performing actual tensor computations
during the export process. This approach results in significantly lower memory usage.

In this example, we use the HighResNet model from MONAI. Before proceeding, please install it from PyPI:

.. code-block:: bash

   pip install monai


PyTorch offers a tool for capturing and visualizing memory usage traces. We will use this tool to record the memory usage of the two
exporters during the export process and compare the results. You can find more details about this tool on
`Understanding CUDA Memory Usage <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_.



TorchDynamo-based exporter
==========================

The code below could be run to generate a snapshot file which records the state of allocated CUDA memory during the export process.

.. code-block:: python

    import torch

    from monai.networks.nets import (
        HighResNet,
    )

    torch.cuda.memory._record_memory_history()

    model = HighResNet(
        spatial_dims=3, in_channels=1, out_channels=3, norm_type="batch"
    ).eval()

    model = model.to("cuda")
    data = torch.randn(30, 1, 48, 48, 48, dtype=torch.float32).to("cuda")

    with torch.no_grad():
        onnx_program = torch.onnx.export(
                            model,
                            data,
                            "test_faketensor.onnx",
                            dynamo=True,
                        )

    snapshot_name = f"torchdynamo_exporter_example.pickle"
    print(f"generate {snapshot_name}")

    torch.cuda.memory._dump_snapshot(snapshot_name)
    print(f"Export is done.")

Open `pytorch.org/memory_viz <https://pytorch.org/memory_viz>`_ and drag/drop the generated pickled snapshot file into the visualizer.
The memory usage is described as below:

.. image:: _static/img/onnx/torch_dynamo_exporter_memory_usage.png


By this figure, we can see the memory usage peak is only around 45MB.

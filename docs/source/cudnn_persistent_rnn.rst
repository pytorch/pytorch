.. note::

    If the following conditions are satisfied:
    1) cudnn is enabled, 
    2) input data is on the GPU 
    3) input data has dtype ``torch.float16`` 
    4) V100 GPU is used,
    5) input data is not in ``PackedSequence`` format
    persistent algorithm can be selected to improve performance.  

from torch.testing import floating_types
from .core import OpInfo


class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""
    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=None,
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 sample_inputs_func=None,
                 **kwargs):
        super(ShapeFuncInfo, self).__init__(name,
                                            dtypes=dtypes,
                                            dtypesIfCPU=dtypesIfCPU,
                                            dtypesIfCUDA=dtypesIfCUDA,
                                            dtypesIfROCM=dtypesIfROCM,
                                            sample_inputs_func=sample_inputs_func,
                                            **kwargs)
        self.ref = ref

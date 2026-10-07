"""PyTorch-owned QuACK EpiOps used by generated FlexGEMM epilogues.

The vendored QuACK tree carries only generic protocol hooks; every concrete
FlexGEMM op lives here and is handed to QuACK as a first-class EpiOp. Import
lazily: these modules load CuTeDSL.
"""

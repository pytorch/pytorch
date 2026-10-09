# NCCL EP source in PyTorch

This directory contains only `nccl_ep/` from the public history of
https://github.com/NVIDIA/nccl-extensions at commit
`ab38d31b10af0247ec0dbabb97e996b6185fda69` (NCCL EP API version 1).

The 45 upstream entries are unchanged. All runtime source and header files
match NCCL commit `73cf112295c33aee2b895f329f592f2a9b4b0f97` under
`contrib/nccl_ep/`. This keeps the source migration separate from an EP upgrade.

The other extensions and upstream NCCL and googletest submodules are not
included. PyTorch supplies the NCCL build through
`cmake/External/nccl_ep_build/CMakeLists.txt`.

`LICENSE.ThirdPartyNotices.txt` is an unchanged copy of upstream
`ThirdPartyNotices.txt`. The extra filename lets PyTorch's license audit and
wheel license metadata include these notices.

To update this snapshot, copy only `nccl_ep/` from the selected upstream
revision, preserve file modes and symlinks, refresh the notices copy, and
update this revision record. Validate both NCCL linkage configurations and
run `test/distributed/test_nccl_ep_build.py` and the TokenSwitch tests.

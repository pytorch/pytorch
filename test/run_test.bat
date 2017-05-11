@echo off

set PYCMD=python

echo Running torch tests
%PYCMD% test_torch.py

echo Running autograd tests
%PYCMD% test_autograd.py

echo Running sparse tests
%PYCMD% test_sparse.py

echo Running nn tests
%PYCMD% test_nn.py

echo Running legacy nn tests
%PYCMD% test_legacy_nn.py

echo Running optim tests
%PYCMD% test_optim.py

echo Running multiprocessing tests
set MULTIPROCESSING_METHOD=""
%PYCMD% test_multiprocessing.py
set MULTIPROCESSING_METHOD=spawn
%PYCMD% test_multiprocessing.py
set MULTIPROCESSING_METHOD=forkserver
%PYCMD% test_multiprocessing.py

echo Running util tests
%PYCMD% test_utils.py

echo Running dataloader tests
%PYCMD% test_dataloader.py

echo Running cuda tests
%PYCMD% test_cuda.py

echo Running NCCL tests
%PYCMD% test_nccl.py


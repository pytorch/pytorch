# not for land
# test harness for vasiliy-debug

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._dynamo.vasiliy_debug_extract_subgraphs import debug_linears_for_float8
from torch._dynamo.vasiliy_debug_analyze_subgraphs import analyze_subgraphs


def test_debug():
    # test, for debugging
    target_folder = '/home/vasiliy/local/tmp/20240802_dynamo_test'
    # note: this test currently requires data to already be in target_folder
    analyze_subgraphs(target_folder, extracted_bsz=4, target_bsz=32)

def test_interformer():
    # real interformer subgraphs
    target_folder = '/home/vasiliy/local/tmp/20240802_interformer_subgraphs'
    analyze_subgraphs(target_folder, extracted_bsz=8, target_bsz=1280)
    # analyze_subgraphs(target_folder, extracted_bsz=8, target_bsz=640)

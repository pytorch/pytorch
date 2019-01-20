#!/bin/bash

#xdoctest -m torch --style=google list

cd $HOME/code/pytorch
xdoctest ./torch --style=google --global-exec "from torch import nn\nimport torch.nn.functional as F\nimport torch" --options=+IGNORE_WHITESPACE

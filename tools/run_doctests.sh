#!/bin/bash

#xdoctest -m torch --style=google list

cd $HOME/code/pytorch
# Note: use freeform until 0.7.3 xdoctest release where I add support for double colon examples
xdoctest ./torch --style=freeform --global-exec "from torch import nn\nimport torch.nn.functional as F\nimport torch" --options=+IGNORE_WHITESPACE

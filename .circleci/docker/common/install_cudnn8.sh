#!/bin/bash

sudo apt-get update
# also install ssh to avoid error of:
# --------------------------------------------------------------------------
# The value of the MCA parameter "plm_rsh_agent" was set to a path
# that could not be found:
#   plm_rsh_agent: ssh : rsh
sudo apt-get install -y ssh
sudo apt-get update && apt-get install -y --no-install-recommends libcudnn8=8.2.0.53-1+cuda11.3 libcudnn8-dev=8.2.0.53-1+cuda11.3 && apt-mark hold libcudnn8

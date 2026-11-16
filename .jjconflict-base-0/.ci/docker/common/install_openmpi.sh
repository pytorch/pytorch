#!/bin/bash

sudo apt-get update
# also install ssh to avoid error of:
# --------------------------------------------------------------------------
# The value of the MCA parameter "plm_rsh_agent" was set to a path
# that could not be found:
#   plm_rsh_agent: ssh : rsh
sudo apt-get install -y ssh
sudo apt-get install -y --allow-downgrades --allow-change-held-packages openmpi-bin libopenmpi-dev

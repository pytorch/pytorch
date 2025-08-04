#!/bin/bash

set -e

rm -rf model
python export.py

unzip model.pt2
echo "model.pt2 is generated and unzipped"

rm -f model.pt2
echo "Remove model.pt2 so that the Windows compilation will create a new model.pt2"

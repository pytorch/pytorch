#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Optional, Dict, Any
import os
import yaml
import torch.jit.operator_upgraders
import torch

# populates the upgrader map to be used in runtime
def main():
    torch.jit.operator_upgraders.generate_graph()

if __name__ == "__main__":
    main()

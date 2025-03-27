#!/usr/bin/env python3
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 运行示例脚本
exec(open('examples/cosmic_optimization_example.py').read()) 
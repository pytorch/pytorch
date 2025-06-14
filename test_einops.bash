#!bin/bash

pip install einops==0.8
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.7
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.6
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.5
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.4
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.3
python test/dynamo/test_einops_interop.py --use-pytest --tb=line

pip install einops==0.2
python test/dynamo/test_einops_interop.py --use-pytest --tb=line
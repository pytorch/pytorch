#!/bin/bash
python -m fastrnns.bench --fuser=old --group=rnns --print-json oss > old.json
python -m fastrnns.bench --fuser=te --group=rnns  --print-json oss > te.json
python compare-fastrnn-results.py old.json te.json --format md

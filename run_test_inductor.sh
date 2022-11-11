#!/bin/bash
set -x
pytest --csv test_torchinductor.csv -v test/inductor/test_torchinductor.py 2>&1 | tee test_torchinductor.log
gh gist create test_torchinductor.log | tee gist.test_torchinductor.csv
python test_extract.py test_torchinductor.csv > final_test_torchinductor.csv
gh gist create final_test_torchinductor.csv

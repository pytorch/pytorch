#!/bin/bash
set -x
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi
export SUPPRESS_XFAILS=1
pytest --csv test_proxy_tensor.csv -v test/test_proxy_tensor.py -k test_make_fx_symbolic_exhaustive 2>&1 | tee test_proxy_tensor.log
gh gist create test_proxy_tensor.log | tee gist.test_proxy_tensor.csv
python test_extract.py test_proxy_tensor.csv > final_test_proxy_tensor.csv
gh gist create final_test_proxy_tensor.csv

#!/bin/bash
set -x
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi
export SUPPRESS_XFAILS=1
pytest --csv test_aotdispatch.csv -v test/functorch/test_aotdispatch.py -k test_aot_autograd_symbolic_exhaustive 2>&1 | tee test_aotdispatch.log
gh gist create test_aotdispatch.log | tee gist.test_aotdispatch.csv
python test_extract.py test_aotdispatch.csv > final_test_aotdispatch.csv
gh gist create final_test_aotdispatch.csv

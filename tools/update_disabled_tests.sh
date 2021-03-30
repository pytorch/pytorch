#!/bin/bash
EXTRACTED_REPO=https://$USERNAME:$API_KEY@github.com/zdevito/pytorch_disabled_tests.git
git clone $EXTRACTED_REPO
cd pytorch_disabled_tests
curl 'https://api.github.com/search/issues?q=is%3Aissue+is%3Aopen+label%3A%22topic%3A+flaky-tests%22+repo:pytorch/pytorch+in%3Atitle+DISABLED' \
| sed 's/"score": [0-9\.]*/"score": 0.0/g' > result.json
# score changes every request, so we strip it out to avoid creating a commit every time we query.
git commit -a -m 'update'
git push

#!/usr/bin/env bash

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)

TMPFILE=$(mktemp)
trap "rm -rf ${TMPFILE}" EXIT

# By default just run against the latest commit
BASE=${BASE:-HEAD~1}
HEAD=${HEAD:-HEAD}

ancestor=$(git merge-base "${BASE}" "${HEAD}")
echo "INFO: Checking aginst the following stats"
(
    set -x
    git diff --stat=10000 "$ancestor" "${HEAD}" | sed '$d' > "${TMPFILE}"
)

while read -r git_attribute; do
    if echo "${git_attribute}" | grep "linguist-generated=true" >/dev/null 2>/dev/null; then
        pattern=$(echo ${git_attribute} | cut -d' ' -f1)
        escaped_pattern=$(printf '%s\n' "$pattern" | sed -e 's/[\/&]/\\&/g')
        # Delete known generated files
        sed -i '/'"${escaped_pattern}"'/d' "${TMPFILE}"
    fi
done < "${GIT_TOP_DIR}/.gitattributes"

echo "INFO: Showing non-generated files:"
(
    set -x
    cat "${TMPFILE}"
)

# Get only files that have changed
changed_files=$(cut -d' ' -f2 "${TMPFILE}" | xargs)

details=$(git diff --shortstat "$ancestor" "${HEAD}" -- ${changed_files})
add=$(echo "$details" | grep -o '[0-9]* insertion' | grep -o '[0-9]*' || true)
remove=$(echo "$details" | grep -o '[0-9]* deletion' | grep -o '[0-9]*' || true)
pr_size=0
if [ "$add" ]; then
  pr_size=$((pr_size + add))
fi
if [ "$remove" ]; then
  pr_size=$((pr_size + remove))
fi
echo "INFO: PR SIZE is ${pr_size}"

if ((pr_size > 2000)); then
    echo
    echo 'Your PR is '"$pr_size"' LOC which is more than the 2000 maximum'
    echo 'allowed within PyTorch infra. PLease make sure to split up'
    echo 'your PR into smaller pieces that can be reviewed.'
    echo 'If you think that this rule should not apply to your PR,'
    echo 'please contact @albanD or @seemethere.'
    echo
    exit 1
fi

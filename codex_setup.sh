set -ex
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install numpy
lintrunner init
NIGHTLY_PATCH=$(curl -s https://github.com/pytorch/pytorch/commit/nightly.patch | head -n20)
COMMIT=$(grep -oE '[0-9a-f]{40}' <<< "$NIGHTLY_PATCH" | head -1)
COMMIT_DATE=$(echo "$NIGHTLY_PATCH" | grep '^Date:' | sed -E 's/Date: .*, ([0-9]+) ([A-Za-z]+) ([0-9]+) .*/\3 \2 \1/' | awk 'BEGIN{split("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec", months, " "); for(i=1;i<=12;i++) month[months[i]]=sprintf("%02d",i)} {print $1 month[$2] sprintf("%02d",$3)}')
VERSION_STRING="2.9.0.dev${COMMIT_DATE}+cpu"
git rev-parse HEAD > /tmp/orig_work.txt
git reset --hard $COMMIT
USE_NIGHTLY=$VERSION_STRING python setup.py develop
echo "source $PWD/.venv/bin/activate" >> ~/.bashrc

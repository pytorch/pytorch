#!/usr/bash
set -x

# pass the path of 
if [[ $# < 1 ]]; then
  echo "Usage: bash logextract_demo.sh [path/to/pytorch/repo]"
  echo ""
  echo "Error: missing path to pytorch repo"
  exit 1
fi

echo "First, we write a test python file to logextract_demo_script.py"
cat > logextract_demo_script.py <<EOF
import torch

def fn(x, y):
    x1 = torch.exp(x)
    y1 = torch.cos(y)
    return x1 + y1

x = torch.rand((400, 400)).cuda()
y = torch.rand((400, 400)).cuda()

# with fuser2 enables nvfuser
with torch.jit.fuser("fuser2"):
    fn_s = torch.jit.script(fn)
    fn_s(x, y)
    print(fn_s(x, y))
EOF

cat logextract_demo_script.py

echo "Next, we run the example python file with PYTORCH_JIT_LOG_LEVEL=>>graph_fuser and pipe the output to logextract_demo_logs.txt"
PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" python logextract_demo_script.py &> logextract_demo_logs.txt

echo "Now we run log_extract."

echo "First, let's just dump the fusion group"
python "${1}/scripts/jit/log_extract.py" logextract_demo_logs.txt --output

echo "Now, try benchmarking against the no-fusion implementation"
python "${1}/scripts/jit/log_extract.py" logextract_demo_logs.txt --baseline --nvfuser

#!/bin/bash
cat > /tmp/git-rebase-todo << 'EOF'
pick 177cc647fa5 add run_graph
pick ebffd343937 display user FX annotations in run_graph()
pick 34085b64007 DebugMode hooks
pick 36abe88f487 Add nn.Module tracking to DebugMode
pick 8773c3ccf48 record inductor runtime calls
EOF

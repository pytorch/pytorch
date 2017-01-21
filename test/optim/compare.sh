
th test.lua > lua.out
python3 test.py > python.out

diff lua.out python.out >/dev/null 2>&1
RESULT=$?
if [[ RESULT -eq 0 ]]; then
    echo "PASS"
else
    echo "FAIL"
    echo "Press ENTER to open vimdiff"
    read
    vimdiff lua.out python.out
fi

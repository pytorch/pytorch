import csv
import sys
import subprocess

out = csv.writer(sys.stdout, dialect='excel')
gist = open('gist.' + sys.argv[1], 'r').read().rstrip()
hash = subprocess.check_output('git rev-parse HEAD'.split(' '), encoding='utf-8').rstrip()

out.writerow([hash, gist, ""])

with open(sys.argv[1], 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'] not in {'failed', 'error'}:
            continue
        msg = row['message'].split('\n')[0]
        msg.replace(" - erroring out! It's likely that this is caused by data-dependent control flow or similar.", "")
        msg.replace("\t", " ")
        out.writerow([row['name'].replace('test_make_fx_symbolic_exhaustive_', ''), msg, ""])

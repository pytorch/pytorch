file1 = open('ts_log.txt', 'r')
Lines = file1.readlines()
  
count = 0
# Strips the newline character
ops = set()
for line in Lines:
    if ": Tensor = aten::" in line:
        s = line.find(": Tensor = aten::")
        e = line.find("(")
        ops.add(line[s+17:e])
print(ops)
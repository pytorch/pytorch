import sys

if len(sys.argv) != 3:
    print("Usage: script.py <arg1> <arg2>")
    sys.exit(1)

arg1 = sys.argv[1]
arg2 = sys.argv[2]

# Example usage of the arguments
import json

try:
    with open(arg1, 'r') as file1:
        data1 = json.load(file1)
    with open(arg2, 'r') as file2:
        data2 = json.load(file2)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    sys.exit(1)
def process(s):
    return s[1:-1].split(',')
lows = []
for cs1, cs2 in zip(data1["collated_samples"], data2["collated_samples"]):
    
    d1_min = 1000000000000.0
    d2_min = 1000000000000.0
    for s1, s2 in zip(cs1, cs2):
        d1 = process(s1)
        d2 = process(s2)
        d1_val = float(d1[-1])
        d2_val = float(d2[-1])
        if d1_val < d1_min:
            d1_min = d1_val
        if d2_val < d2_min:
            d2_min = d2_val
    lows.append((d1_min, d2_min))
diff = 0
i = 0
for d1, d2 in lows:
    i += 1
    diff += (d1 - d2) / d1
print(f"{(diff / i) * 100}%")

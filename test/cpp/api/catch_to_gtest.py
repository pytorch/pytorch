import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('file')
options = parser.parse_args()

with open(options.file) as file:
    source = file.read()


for match in re.finditer(r"TEST\(CursorTest, (.*)\)", source, re.M):
    words = ''.join([w[0].upper() + w[1:].replace(')', '').replace('(', '') for w in match.group(1).split(' ')])
    print("TEST(CursorTest, {}) {{".format(words))

# print(source)

# with open(options.file, 'w') as file:
#     file.write(source)

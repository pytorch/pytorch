import os
import pandas as pd
import subprocess

root_dir = "/dockerx/pytorch/scripts/amd"
data_path = os.path.join(root_dir, "20220318_Daily_Mathews_pytorch_test_audit.xlsx")
df = pd.read_excel(data_path, sheet_name="20220316_ROCm4.5.2_test1_test2", engine='openpyxl')

df = df[df["Assignee"] == "Michael Melesse"]

commands = {}
for d in df.to_dict(orient="records"):
    # print(d)
    test_file = os.path.join("/tmp/pytorch/test", d['test file'])
    test_class = d['class'].replace("(", "").replace(")", "").replace("__main__.", "")
    test_name = d['test name']
    test_log = test_class + "." + test_name
    if test_file not in commands:
        commands[test_file] = "python {}.py --verbose {}.{} ".format(
            test_file, test_class, test_name)
    else:
        commands[test_file] += "{}.{} ".format(test_class, test_name)


for file, command in commands.items():
    print("file", file)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # exit()

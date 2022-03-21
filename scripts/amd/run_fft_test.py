import os
import pandas as pd
import subprocess

bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"


root_dir = "/dockerx/pytorch/scripts/amd"
data_path = os.path.join(root_dir, "20220318_Daily_Mathews_pytorch_test_audit.xlsx")
df = pd.read_excel(data_path, sheet_name="20220316_ROCm4.5.2_test1_test2", engine='openpyxl')

df = df[df["Assignee"] == "Michael Melesse"]

for d in df.to_dict(orient="records"):
    # print(d)
    test_file = os.path.join("/tmp/pytorch/test", d['test file'])
    test_class = d['class'].replace("(", "").replace(")", "").replace("__main__.", "")
    test_name = d['test name']
    test_log = test_class + "." + test_name
    command = "python {}.py --verbose {}.{}".format(
        test_file, test_class, test_name, test_log)
    print(command)

    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # exit()

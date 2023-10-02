import os

def list_files_recursive(path):
    files_list = []

    # Iterate over the items in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Append the full path of each file to the files_list
            files_list.append(os.path.join(root, file))

    return files_list

if __name__ == "__main__":
    path = "/home/eellison/pytorch/torch/_inductor"
    all_files = list_files_recursive(path)

    for file_path in all_files:
        file_path = file_path.replace("/home/eellison/pytorch/", "./")
        if file_path[-3:] == ".py":
            print(file_path)

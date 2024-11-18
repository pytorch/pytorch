from download_reports import download_reports

if __name__ == "__main__":
    commit = "d89d6ed75b70bf4808b9afb4486e540dd490a770"
    dynamo311, eager311 = download_reports(commit, ("dynamo311", "eager311"))
    print("dynamo311:{dynamo311}")
    print("eager311:{eager311}")

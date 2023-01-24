namespace caffe2 {
namespace python {

Workspace* GetCurrentWorkspace();
void SetCurrentWorkspace(Workspace* workspace);
Workspace* NewWorkspace();
Workspace* GetWorkspaceByName(const std::string& name);
std::string GetCurrentWorkspaceName();
void InsertWorkspace(const std::string& name, std::unique_ptr<Workspace> ws);
void SwitchWorkspaceInternal(const std::string& name, bool create_if_missing);
void ResetWorkspace(Workspace* workspace);
void GetWorkspaceNames(std::vector<std::string>& names);
void ClearWorkspaces();
} // namespace python
} // namespace caffe2

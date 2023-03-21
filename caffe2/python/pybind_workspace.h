#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//#include <Python.h>

namespace caffe2 {
namespace python {
class C10_EXPORT BlobFetcherBase {
 public:
  struct FetchedBlob {
    pybind11::object obj;
    bool copied;
  };
  virtual ~BlobFetcherBase();
  virtual pybind11::object Fetch(const Blob& blob) = 0;
};

C10_DECLARE_TYPED_REGISTRY(
    BlobFetcherRegistry,
    TypeIdentifier,
    BlobFetcherBase,
    std::unique_ptr);
#define REGISTER_BLOB_FETCHER(id, ...) \
  C10_REGISTER_TYPED_CLASS(BlobFetcherRegistry, id, __VA_ARGS__)
inline unique_ptr<BlobFetcherBase> CreateFetcher(TypeIdentifier id) {
  return BlobFetcherRegistry()->Create(id);
}

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

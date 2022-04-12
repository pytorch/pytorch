#pragma once
struct THPStorage {
  PyObject_HEAD
  c10::StorageImpl *cdata;
};

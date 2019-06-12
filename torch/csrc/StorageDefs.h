#pragma once
struct THPStorage {
  PyObject_HEAD
  THWStorage *cdata;
};

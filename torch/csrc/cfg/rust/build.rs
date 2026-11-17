fn main() {
    // The Py_3_x cfgs: a few places depend on the layout of CPython's objects.
    pyo3_build_config::use_pyo3_cfgs();
}

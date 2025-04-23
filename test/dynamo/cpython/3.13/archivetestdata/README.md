# Test data for `test_zipfile`, `test_tarfile` (and even some others)

## `test_zipfile`

The test executables in this directory are created manually from `header.sh` and
the `testdata_module_inside_zip.py` file.  You must have Info-ZIP's zip utility
installed (`apt install zip` on Debian).

### Purpose of `exe_with_zip` and `exe_with_z64`

These are used to test executable files with an appended zipfile, in a scenario
where the executable is _not_ a Python interpreter itself so our automatic
zipimport machinery (that'd look for `__main__.py`) is not being used.

### Updating the test executables

If you update header.sh or the testdata_module_inside_zip.py file, rerun the
commands below.  These are expected to be rarely changed, if ever.

#### Standard old format (2.0) zip file

```
zip -0 zip2.zip testdata_module_inside_zip.py
cat header.sh zip2.zip >exe_with_zip
rm zip2.zip
```

#### Modern format (4.5) zip64 file

Redirecting from stdin forces Info-ZIP's zip tool to create a zip64.

```
zip -0 <testdata_module_inside_zip.py >zip64.zip
cat header.sh zip64.zip >exe_with_z64
rm zip64.zip
```

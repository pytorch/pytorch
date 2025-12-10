import os

import pytest

import fsspec


def test_move_raises_error_with_tmpdir(tmpdir):
    # Create a file in the temporary directory
    source = tmpdir.join("source_file.txt")
    source.write("content")

    # Define a destination that simulates a protected or invalid path
    destination = tmpdir.join("non_existent_directory/destination_file.txt")

    # Instantiate the filesystem (assuming the local file system interface)
    fs = fsspec.filesystem("file")

    # Use the actual file paths as string
    with pytest.raises(FileNotFoundError):
        fs.mv(str(source), str(destination))


@pytest.mark.parametrize("recursive", (True, False))
def test_move_raises_error_with_tmpdir_permission(recursive, tmpdir):
    # Create a file in the temporary directory
    source = tmpdir.join("source_file.txt")
    source.write("content")

    # Create a protected directory (non-writable)
    protected_dir = tmpdir.mkdir("protected_directory")
    protected_path = str(protected_dir)

    # Set the directory to read-only
    if os.name == "nt":
        os.system(f'icacls "{protected_path}" /deny Everyone:(W)')
    else:
        os.chmod(protected_path, 0o555)  # Sets the directory to read-only

    # Define a destination inside the protected directory
    destination = protected_dir.join("destination_file.txt")

    # Instantiate the filesystem (assuming the local file system interface)
    fs = fsspec.filesystem("file")

    # Try to move the file to the read-only directory, expecting a permission error
    with pytest.raises(PermissionError):
        fs.mv(str(source), str(destination), recursive=recursive)

    # Assert the file was not created in the destination
    assert not os.path.exists(destination)

    # Cleanup: Restore permissions so the directory can be cleaned up
    if os.name == "nt":
        os.system(f'icacls "{protected_path}" /remove:d Everyone')
    else:
        os.chmod(protected_path, 0o755)  # Restore write permission for cleanup

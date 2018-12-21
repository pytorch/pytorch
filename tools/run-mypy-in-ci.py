import subprocess
import sys


if __name__ == '__main__':
    # get files in changeset
    files_changed = subprocess.check_output(['git', 'diff', '--name-only', 'master'])
    files_changed = files_changed.decode(sys.getfilesystemencoding())
    files_changed = files_changed.split("\n")
    files_changed = list(filter(lambda f: f.endswith('.py'), files_changed))

    def is_relevant_error(line):
        return True

    command = ['mypy'] + files_changed
    print(" ".join(command))
    return_code = 0
    output = None
    try:
        subprocess.check_output(command, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        return_code = err.returncode
        output = err.output.decode(sys.getfilesystemencoding())

    if return_code == 0:
        # no errors in these files
        sys.exit(0)

    # strip errors not introduced in this changeset
    errors = output.split("\n")
    errors = list(filter(lambda line: is_relevant_error(line), errors))

    if len(errors) == 0:
        # no errors introduced in this changeset, everything is fine
        sys.exit(0)

    print("\n".join(errors))
    sys.exit(return_code)

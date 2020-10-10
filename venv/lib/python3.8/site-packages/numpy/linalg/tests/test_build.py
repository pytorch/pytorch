from subprocess import PIPE, Popen
import sys
import re
import pytest

from numpy.linalg import lapack_lite
from numpy.testing import assert_


class FindDependenciesLdd:

    def __init__(self):
        self.cmd = ['ldd']

        try:
            p = Popen(self.cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
        except OSError:
            raise RuntimeError("command %s cannot be run" % self.cmd)

    def get_dependencies(self, lfile):
        p = Popen(self.cmd + [lfile], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if not (p.returncode == 0):
            raise RuntimeError("failed dependencies check for %s" % lfile)

        return stdout

    def grep_dependencies(self, lfile, deps):
        stdout = self.get_dependencies(lfile)

        rdeps = dict([(dep, re.compile(dep)) for dep in deps])
        founds = []
        for l in stdout.splitlines():
            for k, v in rdeps.items():
                if v.search(l):
                    founds.append(k)

        return founds


class TestF77Mismatch:

    @pytest.mark.skipif(not(sys.platform[:5] == 'linux'),
                        reason="no fortran compiler on non-Linux platform")
    def test_lapack(self):
        f = FindDependenciesLdd()
        deps = f.grep_dependencies(lapack_lite.__file__,
                                   [b'libg2c', b'libgfortran'])
        assert_(len(deps) <= 1,
                         """Both g77 and gfortran runtimes linked in lapack_lite ! This is likely to
cause random crashes and wrong results. See numpy INSTALL.txt for more
information.""")

import subprocess, sys, textwrap, tempfile, pathlib, os

def test_inductor_counters_are_quiet_by_default():
    code = textwrap.dedent(
        '''
        import logging, torch, unittest
        from torch._inductor.test_case import TestCase
        logging.basicConfig(level=logging.WARNING)

        class MyCase(TestCase):
            def test_compile(self):
                f = torch.compile(lambda x: x.relu())
                f(torch.randn(2, 2))

        if __name__ == "__main__":
            unittest.main(verbosity=0)
        '''
    )
    tmp = pathlib.Path(tempfile.mkdtemp()) / "repro.py"
    tmp.write_text(code)

    proc = subprocess.run(
        [sys.executable, str(tmp)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        print("STDERR from child:\n", proc.stderr, file=sys.stderr)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""

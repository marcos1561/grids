import unittest
import subprocess
import sys
import os
import pathlib

class TestExamplesRun(unittest.TestCase):
    def test_all_examples_run(self):
        examples_dir = pathlib.Path(__file__).parent.parent / "examples"
        for script in examples_dir.glob("*.py"):
            with self.subTest(script=script):
                env = os.environ.copy()
                env["MATPLOTLIB_BACKEND"] = "Agg"
                result = subprocess.run([sys.executable, str(script)], timeout=30, env=env)
                self.assertEqual(result.returncode, 0, f"{script} crashed with exit code {result.returncode}")

if __name__ == "__main__":
    unittest.main()
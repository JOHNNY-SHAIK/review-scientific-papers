import unittest
import subprocess
import sys
import os

class TestDataCheckScript(unittest.TestCase):
    def test_sample_dataset_passes(self):
        script = os.path.join("src", "data_check.py")
        sample = os.path.join("data", "mldataset_sample.json")
        # run script as subprocess and capture output
        proc = subprocess.run([sys.executable, script, sample], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"Script failed with return code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertIn("All records passed basic validation.", proc.stdout)

if __name__ == "__main__":
    unittest.main()

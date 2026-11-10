#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PY = os.path.join(ROOT_DIR, "app.py")

def main():
    with tempfile.TemporaryDirectory() as tmp_out:
        cmd = [sys.executable, APP_PY, "--query-id", "q_mi300x_fp16", "--query", "What is the Peak FP16 / BF16 Matrix TFLOPS (without sparsity) for AMD Instinct MI300X?", "--output-dir", tmp_out]
        subprocess.run(cmd, check=True, timeout=15)
        with open(os.path.join(tmp_out, "q_mi300x_fp16_output.json"), "r", encoding="utf-8") as f:
            r1 = json.load(f)
        assert r1["answer"] == "1,307.4 TFLOPS", f"Unexpected: {r1}"
        assert r1["verified_substring"] is True
        print("MATCH 4 HARNESS PASSED:", r1["answer"])

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import time
from extractor import VeriFetchExtractor

def main():
    p = argparse.ArgumentParser(description="VeriFetch ROCm Match 4 CLI Entrypoint")
    p.add_argument("--query", "-q", "--question", dest="query", default="")
    p.add_argument("--query-id", "--id", dest="query_id", default="q_default")
    p.add_argument("--url", "-u", default="")
    p.add_argument("--corpus", "-c", default="")
    p.add_argument("--html", default="")
    p.add_argument("--output", "-o", default="")
    p.add_argument("--output-dir", default="/app/output")
    args = p.parse_args()

    out_dir = args.output_dir if os.access("/", os.W_OK) else "output"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    res = VeriFetchExtractor().extract(args.query_id, args.query, args.url, args.corpus, args.html)
    res["latency_sec"] = round(time.perf_counter() - t0, 4)
    out_path = args.output or os.path.join(out_dir, f"{args.query_id}_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

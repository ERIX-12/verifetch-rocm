#!/usr/bin/env python3
"""
VeriFetch ROCm — Persistent Unix Socket Daemon (/app/daemon_rocm.py)
Keeps the extractor and local AMD ROCm model warm in HBM3 VRAM behind /tmp/verifetch.sock.
"""
import json
import os
import socket
from extractor import VeriFetchExtractor

SOCKET_PATH = os.environ.get("VERIFETCH_SOCKET", "/tmp/verifetch.sock")

def serve_forever():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    extractor = VeriFetchExtractor()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o666)
    server.listen(16)
    print(f"[VeriFetch ROCm Daemon] Warm and listening on {SOCKET_PATH}")
    while True:
        conn, _ = server.accept()
        try:
            raw = conn.recv(262144).decode("utf-8").strip()
            if raw:
                req = json.loads(raw)
                res = extractor.extract(
                    query_id=req.get("query_id", "q_0"),
                    query=req.get("query", ""),
                    url=req.get("url", ""),
                    corpus_dir=req.get("corpus_dir", ""),
                    html_override=req.get("html_override", ""),
                )
                conn.sendall(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            conn.sendall(json.dumps({"error": str(exc), "answer": "", "citations": []}).encode("utf-8"))
        finally:
            conn.close()

if __name__ == "__main__":
    serve_forever()

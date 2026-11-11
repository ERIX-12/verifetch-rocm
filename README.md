# VeriFetch ROCm Suite — AMD AI League (Matches 2, 3 & 4 on AMD Instinct MI300X)

Built by **Tusiime Eriya** (`ERIX-12` · Team `VeriFetch ROCm`) for the **AMD AI League** on `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0`.

## Multi-Repository Index (Matches 2, 3 & 4)
Per the AMD AI League submission guidelines (*"If you have multiple repositories, mention them in the readme"*), this project spans three dedicated `rocm/pytorch` containers:

1. **Match 4 — Intelligent Web Information Retrieval (`+1,800 XP`):**
   - **Repository:** `ERIX-12/verifetch-rocm` (This Repository)
   - **Architecture:** Sub-second (`0.42s`) warm Unix-socket daemon (`daemon_rocm.py`), structure-preserving HTML `<table>` linearizer (`extractor.py`), and deterministic `Cite-or-Refuse` substring verifier (`0.0%` hallucination rate).
2. **Match 3 — Retrieval-Augmented Generation (`+1,500 XP`):**
   - **Repository:** `ERIX-12/rocm-corpus-rag`
   - **Architecture:** Offline mixed-document RAG engine (`rag_engine.py`) parsing PDF, DOCX, XLSX, CSV, Markdown, and logs with hybrid BM25 + dense BGE scoring and exact source filename attribution (`sources` / `files` in `/app/output/<query-id>_output.json`).
3. **Match 2 — Optical Character Recognition (`+1,200 XP`):**
   - **Repository:** `ERIX-12/rocm-vision-ocr`
   - **Architecture:** Resident `Qwen/Qwen2.5-VL-7B-Instruct` (`bfloat16`) VLM OCR engine (`ocr_engine.py`) with adaptive ROI contrast normalization and deterministic alphanumeric post-filtering for license plates, road signs, and industrial labels.

---

## Match 4 Quick Start (`verifetch-rocm`)
```bash
python3 tests/test_match4_harness.py
python3 app.py --query-id q_mi300x_fp16 --query "What is the Peak FP16 / BF16 Matrix TFLOPS (without sparsity) for AMD Instinct MI300X?"
```

## Match 3 Quick Start (`rocm-corpus-rag`)
```bash
python3 app.py --query-id rag_001 --corpus sample_docs --query "How much HBM3 memory does the MI300X have?"
```

## Match 2 Quick Start (`rocm-vision-ocr`)
```bash
python3 app.py --query-id ocr_001 --image sample_plate.png
```

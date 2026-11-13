FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV VERIFETCH_SOCKET=/tmp/verifetch.sock
ENV VERIFETCH_OUTPUT_DIR=/app/output
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir beautifulsoup4 lxml && \
    pip install --no-cache-dir --no-deps -U transformers tokenizers safetensors && \
    python3 -c "import torch; print(torch.__version__)" && \
    mkdir -p /app/output /app/corpus /models
COPY . /app
RUN chmod +x /app/app.py /app/daemon_rocm.py
CMD ["python3", "/app/daemon_rocm.py"]

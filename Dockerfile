FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
WORKDIR /app
RUN pip install --no-cache-dir beautifulsoup4 lxml && mkdir -p /app/output
COPY . /app
CMD ["python3", "/app/app.py", "--help"]

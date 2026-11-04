FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110,<1" "httpx>=0.27,<1" "python-multipart>=0.0.9" "uvicorn[standard]==0.34.0"
COPY main.py .
USER 10001
EXPOSE 8090
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]

# Inference service image (the workload deployed to Kubernetes).
# Run `python -m training.train` first so models/model.pkl exists.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY inference ./inference
COPY models ./models
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]

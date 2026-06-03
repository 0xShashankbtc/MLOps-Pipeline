FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    torch \
    huggingface_hub

COPY src/inference.py .
COPY id2label.json .

ENV MODEL_REPO=G25AIT2100/sms-spam-distilbert

CMD ["python", "inference.py"]

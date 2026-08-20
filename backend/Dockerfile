FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY backend/requirements.txt /tmp/build/
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /tmp/build/requirements.txt

COPY backend/app ./app
COPY backend/knowledge ./knowledge
COPY domains ./domains

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

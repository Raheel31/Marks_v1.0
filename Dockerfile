
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

COPY ./data/processed ./data/processed

RUN mkdir -p /app/logs

CMD ["python", "-m", "uvicorn", "src.main:Marks_v1.0-1", "--host", "0.0.0.0", "--port", "8000"]

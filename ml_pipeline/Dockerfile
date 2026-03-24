FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/train_pipeline.py --samples 60000

EXPOSE 5001

CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5001", \
     "--workers", "2", "--timeout", "120", \
     "api.app:create_app()"]

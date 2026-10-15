FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN git init \
    && git config user.email "render@deploy.local" \
    && git config user.name "Render Deploy" \
    && git add -A \
    && git commit -m "render snapshot" --allow-empty

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# gunicorn (not `python app.py`) is what Render runs in production.
# --preload loads app.py once in the master (model included), then forks a single
# worker so the VGG16 model lives in memory once and is reused across requests.
CMD ["gunicorn", "--preload", "--workers", "1", "--threads", "4", \
     "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]

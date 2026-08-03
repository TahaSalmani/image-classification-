FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN git init \
    && git config user.email "render@deploy.local" \
    && git config user.name "Render Deploy" \
    && git add -A \
    && git commit -m "render snapshot" --allow-empty

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python3", "app.py"]
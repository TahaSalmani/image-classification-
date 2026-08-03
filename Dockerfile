FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN if [ ! -d .git ]; then \
      git init -q && \
      git add -A -q && \
      git -c user.email="render@deploy.local" -c user.name="Render Deploy" commit -q -m "render snapshot" --allow-empty; \
    fi

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python3", "app.py"]
FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python3", "app.py"]
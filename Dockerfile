FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

RUN dvc remote modify origin --local user "${DAGSHUB_USERNAME}"
RUN dvc remote modify origin --local password "${DAGSHUB_TOKEN}" || true

CMD ["python3", "app.py"]
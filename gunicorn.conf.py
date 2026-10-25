import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", "2"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"


def post_worker_init(worker):
    from app import get_model

    try:
        get_model()
        print("model loaded in worker", flush=True)
    except Exception as error:
        print(f"WARNING: model not loaded: {error}", flush=True)

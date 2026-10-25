# image-classification-

# Lung Disease (Pneumonia) Detection

🔗 **Live Demo:** https://image-classification-hm8d.onrender.com

A deep-learning image classification project that detects **Pneumonia** vs **Normal** lungs from chest X-ray images, built with a VGG16 transfer-learning model, orchestrated as a DVC pipeline, and served through a Flask web app.

> Research demo only. This is not a medical device and must not be used for clinical decisions.

## Pipeline

```
config/config.yaml + params.yaml
        |
        v
Stage 1  data_ingestion      Kaggle ZIP -> artifacts/data_ingestion/{train,test,val} + manifest.json
Stage 2  prepare_base_model  VGG16 (ImageNet, no top) -> GlobalAveragePooling + Dense(2)
Stage 3  training            patient-grouped split, augmentation, class weights -> best checkpoint -> model.h5
Stage 4  evaluation          official test split -> scores.json (accuracy, AUC, precision, recall, F1)
        |
        v
DVC (dvc.lock + DagsHub remote) -> Flask + gunicorn -> Docker
```

The ingestion stage verifies the archive with `zipfile.testzip()`, optionally checks a SHA256
(`data_ingestion.verify_sha256`), extracts atomically, rejects unsafe archive paths, validates the
exact class counts in `config/config.yaml`, and writes `manifest.json` before swapping the splits
into place. After the first successful run, copy `archive_sha256` from `manifest.json` into
`config.yaml` to pin the download. `keep_zip` defaults to `false`, so the archive is removed after
extraction. Test and validation splits are DVC-tracked, and evaluation uses the official test set.
Training does not use the 16-image `val/` split (it is too small to measure anything); stage 3
builds a patient-grouped validation set from `train/` instead.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> TensorFlow 2.15.1 (pinned in `requirements.txt`) requires **Python 3.9–3.11**. Python 3.12+
> ships Keras 3 and cannot run this codebase as-is, so create the virtualenv with Python 3.11.

Data ingestion requires the Kaggle CLI and credentials at `~/.kaggle/kaggle.json`:

```bash
dvc repro          # runs all four stages
# or run a single stage:
python src/cnnclassifier/pipeline/stage_01_data_ingestion.py
```

Model and data artifacts live on DagsHub. Configure the remote credentials once
(they are stored locally, not in Git):

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USERNAME"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"

dvc pull
```

After a successful `dvc repro`, publish the new data/model artifacts with `dvc push`.

## Serving

```bash
python app.py
# or production-style:
gunicorn -c gunicorn.conf.py app:app
```

Endpoints:

- `GET /` — web UI
- `GET /health` — liveness and model status
- `POST /predict` — JSON body `{"image": "<base64>"}`; returns the class, confidence, and per-class probabilities

Environment variables:

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path (default `artifacts/training/model.h5`) |
| `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` | Credentials for the startup `dvc pull` |
| `PULL_MODEL_ON_START` | Set to `0` to skip the startup pull |
| `PORT` | Port for gunicorn or the Flask dev server (default `8080`) |
| `GUNICORN_THREADS`, `GUNICORN_TIMEOUT` | gunicorn tuning (defaults `2` and `180`) |

## Docker

```bash
docker build -t cnn-xray .
docker run --rm -p 8080:8080 cnn-xray
```

If PyPI is slow or blocked from your network, point the build at a mirror (the default is
`https://pypi.org/simple`):

```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t cnn-xray .
```

The image installs the serving-only dependency set from `requirements-docker.txt`, so training
and notebook tooling do not inflate the runtime image. It runs gunicorn without `--preload`; the
model is loaded once per worker in `post_worker_init`. This avoids the TensorFlow fork deadlock
caused by preloading.

At startup the container pulls the model from DagsHub before gunicorn binds the port. Give the
service enough time for that pull, or bake/mount the model at `MODEL_PATH` and set
`PULL_MODEL_ON_START=0`. A real VGG16 model plus TensorFlow does not fit comfortably in a 512 MB
instance; use at least 1 GB of RAM and keep `workers = 1`.

## Important

The previous model artifacts were trained with plain `1/255` rescaling. Training, evaluation, and
prediction now all use `tf.keras.applications.vgg16.preprocess_input`, so the model must be
retrained (and the DVC artifacts regenerated) before deploying the new pipeline.

The old artifacts are also not a usable baseline: `dvc.lock` records only **192 training images**,
5 epochs, LR 0.01, and a fully frozen backbone. The current model on DagsHub and the live demo
were produced by that run, so poor predictions from them are expected. Evaluate any new model on
the official 624-image `test/` split (`scores.json`) before deploying it.

The stale `dvc.lock` from the pre-fix pipeline was removed, and `scores.json` is generated by
stage 4. Run `dvc repro` once with Kaggle credentials to regenerate both; the notebooks under
`research/` are historical experiments and are not used by the pipeline.

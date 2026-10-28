# Chest X-ray Pneumonia Classifier

A VGG16 transfer-learning model that classifies chest X-ray images as **Normal** or **Pneumonia**, served through a Flask web app and deployed as a Docker image.

🔗 **Live demo:** https://image-classification-hm8d.onrender.com

> **Research demo only.** This is not a medical device. Predictions can be wrong and must never be used for clinical decisions.

---

## Highlights

- **92% accuracy / 0.984 AUC** on the official 624-image test split
- Patient-grouped validation split, so the same patient never leaks between train and validation
- Class weighting to counter the 1:2.9 NORMAL-to-PNEUMONIA imbalance (98.5% pneumonia sensitivity)
- Reproducible 4-stage DVC pipeline (ingestion → base model → training → evaluation)
- Preprocessing is `tf.keras.applications.vgg16.preprocess_input` everywhere: training, evaluation and serving
- Slim 58.9 MB Keras 3 model baked into the Docker image — no external artifact store needed at runtime

## Results

Evaluated on the official `test/` split (624 images: 234 NORMAL, 390 PNEUMONIA), which is never seen during training or validation:

| Metric | Value |
|---|---|
| Accuracy | 0.920 |
| AUC | 0.984 |
| Precision | 0.924 |
| Recall | 0.920 |
| F1 | 0.918 |
| Loss | 0.223 |

Confusion matrix (rows = true `NORMAL`, `PNEUMONIA`; columns = predicted):

|  | Normal | Pneumonia |
|---|---|---|
| **Normal** | 190 | 44 |
| **Pneumonia** | 6 | 384 |

The model catches 384 of 390 pneumonia cases; the remaining errors are 44 false positives, which is the safer failure direction for a screening aid.

## How it works

```
                     ┌──────────────────────────────┐
  Kaggle archive ──▶ │ 1. data_ingestion            │  train / test / val + manifest.json
                     └──────────────────────────────┘
                     ┌──────────────────────────────┐
  ImageNet weights ─▶│ 2. prepare_base_model        │  VGG16 (no top) + GAP + Dropout + Dense(2)
                     └──────────────────────────────┘
                     ┌──────────────────────────────┐
                     │ 3. training                  │  frozen first 15 layers, augmentation,
                     │                              │  class weights, best-val checkpoint
                     └──────────────────────────────┘
                     ┌──────────────────────────────┐
  624-image test ──▶ │ 4. evaluation                │  scores.json
                     └──────────────────────────────┘
                                │
                                ▼
                 Flask + gunicorn  ──▶  Docker  ──▶  Render
```

**Model:** `VGG16(include_top=False, weights="imagenet")` with the first 15 layers frozen, then `GlobalAveragePooling2D → Dropout(0.3) → Dense(2, softmax)`. 14.7M parameters, 224×224×3 input, Adam at 1e-4, batch 16, 10 epochs, categorical cross-entropy.

**Data:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,875 images (train 1,341 NORMAL / 3,875 PNEUMONIA, test 234 / 390, val 8 / 8). The 16-image `val/` split is too small to measure anything, so training builds a patient-grouped 20% validation set out of `train/` instead, and early stopping restores the best weights.

## Project structure

```
├── app.py                      Flask app: /, /health, /predict
├── gunicorn.conf.py            gunicorn config, loads the model in post_worker_init
├── Dockerfile                  serving image (python:3.11-slim + requirements-docker.txt)
├── config/config.yaml          paths and expected dataset counts
├── params.yaml                 image size, epochs, batch size, LR, freeze depth
├── dvc.yaml                    the four pipeline stages with deps/outs/metrics
├── artifacts/training/model.h5 committed serving model (58.9 MB, Keras 3)
├── knowledge/                  sourced pneumonia passages, SOURCE: on line 1
├── templates/index.html        single-page UI
├── scripts/rag_search.py       query the knowledge base from the CLI
├── scripts/advice_preview.py   print the three advice cards
├── tests/                      knowledge-base, retriever and advice tests
├── src/cnnclassifier/
│   ├── advice_logic.py         three fixed advice modes + LLM fallback text
│   ├── components/             data_ingestion, prepare_base_model, training,
│   │                           prepare_callbacks, evaluation
│   ├── pipeline/               stage_01…stage_04 + predict.py (inference)
│   ├── rag/                    chunker (markdown → passages) + TF-IDF retriever
│   ├── config/, entity/        typed configuration objects
│   └── utils/common.py         yaml/json helpers, base64 image decoding
└── research/                   historical notebooks, not used by the pipeline
```

## Quick start

### Docker (recommended)

```bash
docker build -t chest-xray .
docker run --rm -p 8080:8080 chest-xray
```

Open http://localhost:8080 and upload a chest X-ray. The model is already inside the image, so the container needs no credentials and no network access at startup.

The first build downloads ~1 GB of wheels; if PyPI is slow or blocked on your network, point it at a mirror:

```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t chest-xray .
```

### Local Python

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py            # or: gunicorn -c gunicorn.conf.py app:app
```

`requirements.txt` pins TensorFlow 2.20 (Keras 3) — the version the committed model was trained on — and needs **Python 3.9–3.13**.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness + whether the model is loaded |
| `POST` | `/predict` | Classify one image |

`/health`:

```json
{"model_loaded": true, "model_path": "artifacts/training/model.h5", "status": "ok"}
```

`/predict` takes a JSON body with a base64 string (raw or `data:` URI, up to 10 MB decoded):

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 scan.jpeg)\"}"
```

```json
[
  {
    "image": "Pneumonia",
    "confidence": 0.841,
    "probabilities": {"Normal": 0.159, "Pneumonia": 0.841}
  }
]
```

Uploaded images are written to a temp file, scored, and deleted immediately.

## Knowledge base and retrieval

`knowledge/` holds twelve short, sourced passages about pneumonia — symptoms, emergency and
urgent-care criteria, self-care, treatment, causes, prevention, risk factors, diagnosis, and the
fact that a normal-looking result does not rule pneumonia out. Every file begins with
`SOURCE: <url>` and ends with the licence or attribution line, and the text is quoted from that
page rather than written from memory. The sources are NHS, CDC, and MedlinePlus (NLM); the
MedlinePlus A.D.A.M. encyclopedia entries are copyrighted and deliberately not used.

Retrieval (stage 2 of the planned assistant) is plain TF-IDF from scikit-learn, not an embedding
model: the corpus is ~20 passages, queries take well under a millisecond, and it costs about
30 MB of RAM next to TensorFlow — the difference between fitting and not fitting on a 512 MB
instance. `sentence-transformers` can replace it later behind the same interface.

```bash
python scripts/rag_search.py "when should I go to hospital"     # ranked passages + source URLs
python scripts/rag_search.py --json -k 5 "can I take cough medicine"
python -m unittest discover -s tests -v                          # 14 checks, no TensorFlow needed
```

`TfidfVectorizer` here uses a custom `analyzer` that lowercases, merges `x-ray`/`x ray`/`xray`,
drops English stopwords, strips common suffixes, and emits unigrams plus bigrams. Without the
suffix stripper, "what causes pneumonia" misses the causes passage because it says "caused".

## Advice modes

`src/cnnclassifier/advice_logic.py` turns a prediction into one of three fixed cards, and it runs
with no network, no model API and no key:

| Mode | Condition | Message |
|---|---|---|
| `likely_pneumonia` | p ≥ 0.75 | seek care today; the score is not a diagnosis |
| `uncertain` | 0.40 ≤ p < 0.75 | the score carries no weight either way |
| `likely_normal` | p < 0.40 | a negative score is not a clean bill of health |

Every quoted sentence and every "what to do" line is copied from a `knowledge/` file and tagged
with the file it came from; a test fails if any of them stops matching the source verbatim. The
emergency red flags are shown in all three modes. When the optional language-model explanation is
missing, failed or rate-limited, the card falls back to a fixed sentence and the rest of the page
is unaffected.

The thresholds are a product decision, not a validated operating point — the model has no
calibration study behind it.

```bash
python scripts/advice_preview.py            # all three cards
python scripts/advice_preview.py -p 0.55    # one card, --json for machine output
```

## Training

The pipeline is DVC-orchestrated:

```bash
dvc repro                        # all four stages
python src/cnnclassifier/pipeline/stage_01_data_ingestion.py   # a single stage
```

Stage 1 downloads the Kaggle dataset (needs `~/.kaggle/kaggle.json`), verifies the archive, extracts it atomically, validates the class counts against `config/config.yaml` and writes `manifest.json`. Stage 4 writes `scores.json` with accuracy, AUC, precision, recall, F1 and the confusion matrix.

**Retraining happens on Kaggle.** The current model was trained in a standalone Kaggle notebook (TF 2.20 / Keras 3, 2×T4) and then committed here: the original 110 MB `.h5` was re-saved with `include_optimizer=False` to 58.9 MB so it fits under GitHub's file-size limit, with bit-identical predictions. Two practical notes:

- The legacy stages 3–4 in `src/` still use Keras 2 `ImageDataGenerator`; running `dvc repro` locally requires `tensorflow-cpu==2.15.1` on Python 3.11. The notebook is the supported path for new training runs.
- When you export a model for serving, strip the optimizer, then confirm `/predict` still returns the same probabilities for a known image before deploying.

Artifacts (dataset, older models) are tracked with DVC against a DagsHub remote. Credentials are stored locally, never in Git:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USERNAME"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"
dvc pull
```

## Configuration

Environment variables read by the app:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `artifacts/training/model.h5` | Model location |
| `PULL_MODEL_ON_START` | `1` | `dvc pull` the model at startup if it is missing; set `0` when the model is baked or mounted |
| `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` | — | Credentials for that startup pull |
| `PORT` | `8080` | Port for gunicorn / the dev server |
| `GUNICORN_THREADS` | `2` | gunicorn threads |
| `GUNICORN_TIMEOUT` | `180` | gunicorn worker timeout |

Hyperparameters live in `params.yaml` (`FREEZE_TILL`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `IMAGE_SIZE`), dataset paths and expected counts in `config/config.yaml`.

## Docker & deployment

- `requirements-docker.txt` is a serving-only subset (no Kaggle, notebooks, matplotlib, scikit-learn), keeping the image small.
- gunicorn runs with `workers = 1` and `preload_app = False`; the model loads once in `post_worker_init` to avoid the TensorFlow fork deadlock that `--preload` causes.
- `.dockerignore` excludes everything under `artifacts/` except the serving model, so the image ships the model but no datasets or checkpoints.
- On Render, pushing to `main` builds and deploys automatically. Give the service **at least 1 GB RAM** — VGG16 plus TensorFlow does not fit comfortably in 512 MB.

## Limitations

- Binary NORMAL/PNEUMONIA classification only — no other thoracic findings, no localization, no severity estimate.
- Trained on a single pediatric dataset (Guangzhou Women and Children's Medical Center); performance on adult or out-of-distribution scans will be lower.
- 44 of 234 normal test images are flagged as pneumonia, so a positive result is a prompt for expert review, not a diagnosis.

## License

Released for educational and research use. The dataset carries its own license — check the Kaggle page before redistributing the images.

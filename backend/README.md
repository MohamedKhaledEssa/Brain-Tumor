# Brain Tumor Detection — Backend

FastAPI service that classifies brain MRI images into one of four classes:

| Index | Label        | Description                                                  |
| ----- | ------------ | ------------------------------------------------------------ |
| 0     | `glioma`     | Tumor originating in the glial cells of the brain or spine.  |
| 1     | `meningioma` | Tumor arising from the meninges surrounding the brain.       |
| 2     | `notumor`    | No tumor detected.                                           |
| 3     | `pituitary`  | Tumor of the pituitary gland.                                |

The API powers the Flutter mobile app in `../mobile`.

> **Disclaimer.** This service is for research and educational use only.
> It is **not** a medical device and must not be used for diagnosis. Always
> consult a qualified clinician.

## Layout

```
backend/
├── app/
│   ├── main.py             FastAPI app + lifespan loads the model
│   ├── config.py           env-driven Settings
│   ├── schemas.py          Pydantic request/response models
│   ├── errors.py           HTTP exception helpers
│   ├── ml/
│   │   ├── classes.py      class names + descriptions
│   │   ├── preprocess.py   PIL → tensor pipeline
│   │   └── model.py        TumorDetector wrapper (load weights, predict)
│   └── routers/
│       ├── health.py       GET /, GET /health
│       ├── model_info.py   GET /classes, GET /model/info
│       └── predict.py      POST /predict, POST /predict/batch
├── scripts/train.py        Standalone training entrypoint
├── tests/                  pytest suite (uses FastAPI TestClient)
├── requirements*.txt
├── pyproject.toml
└── Dockerfile
```

## Quickstart

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs> for interactive Swagger UI.

### Demo vs trained mode

The API works **out of the box** without any trained weights — it spins up a
ResNet18 with a randomly initialised 4-class head so you can exercise every
endpoint. `GET /model/info` returns `weights_loaded: false` in this state.

To serve real predictions, train the model and drop the resulting
`brain_tumor.pth` into `backend/models/` (or set `MODEL_PATH`). The next
restart picks them up automatically.

## Endpoints

| Method | Path             | Description                                   |
| ------ | ---------------- | --------------------------------------------- |
| GET    | `/`              | API info.                                     |
| GET    | `/health`        | Liveness probe.                               |
| GET    | `/classes`       | Supported tumor classes + descriptions.       |
| GET    | `/model/info`    | Architecture, device, weights status.         |
| POST   | `/predict`       | Single MRI image upload (`multipart/form-data`, field `file`). |
| POST   | `/predict/batch` | Multiple images (`multipart/form-data`, repeated field `files`). |

### Example

```bash
curl -F "file=@my_scan.jpg" http://localhost:8000/predict
```

```json
{
  "filename": "my_scan.jpg",
  "predicted_label": "glioma",
  "predicted_index": 0,
  "confidence": 0.78,
  "probabilities": [
    {"label": "glioma", "probability": 0.78},
    {"label": "meningioma", "probability": 0.10},
    {"label": "notumor", "probability": 0.05},
    {"label": "pituitary", "probability": 0.07}
  ],
  "inference_ms": 42.1
}
```

## Configuration

All settings are environment variables; see [`.env.example`](.env.example).

| Variable                  | Default                              | Notes                          |
| ------------------------- | ------------------------------------ | ------------------------------ |
| `MODEL_PATH`              | `./models/brain_tumor.pth`           | Trained weights file.          |
| `MODEL_ARCH`              | `resnet18`                           | `resnet18` / `resnet50` / `mobilenet_v3_small`. |
| `DEVICE`                  | `auto`                               | `auto` / `cpu` / `cuda`.       |
| `MAX_UPLOAD_BYTES`        | `10485760`                           | 10 MB.                         |
| `MAX_BATCH_SIZE`          | `16`                                 | Files per batch request.       |
| `CORS_ORIGINS`            | `*`                                  | Comma-separated list.          |
| `USE_PRETRAINED_BACKBONE` | `false`                              | Use ImageNet-pretrained init.  |

## Training

The expected dataset layout matches the
[Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset):

```
<data-root>/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Then run:

```bash
python -m scripts.train \
  --data-root /path/to/Brain_Tumor_MRI_Dataset \
  --arch resnet18 \
  --epochs 10 \
  --pretrained
```

The best-performing checkpoint (by test accuracy) is written to
`backend/models/brain_tumor.pth`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use FastAPI's `TestClient` and synthetic Pillow images, so they don't
require a real dataset and run on CPU only.

## Linting

```bash
ruff check .
```

## Docker

```bash
docker build -t brain-tumor-backend .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  brain-tumor-backend
```

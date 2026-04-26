# Brain Tumor Detection

A two-part project for classifying brain MRI scans:

- **`backend/`** — FastAPI service (Python 3.10+) that serves predictions over
  HTTP. See [`backend/README.md`](backend/README.md) for run, train, and API
  docs.
- **`mobile/`** — Flutter client that uploads scans to the backend and
  displays the result.

## Classes

The model classifies an MRI image into one of four classes:

| Index | Label        |
| ----- | ------------ |
| 0     | `glioma`     |
| 1     | `meningioma` |
| 2     | `notumor`    |
| 3     | `pituitary`  |

## Quickstart

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
# open http://localhost:8000/docs
```

## Disclaimer

This project is for research and educational use only. It is **not** a medical
device. Do not use it for diagnosis. Always consult a qualified clinician.

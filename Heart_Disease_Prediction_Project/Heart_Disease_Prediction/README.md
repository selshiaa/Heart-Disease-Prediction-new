# Heart Disease Prediction System

Classic, professional frontend with a Python machine-learning backend.

## Project structure

```text
Heart_Disease_Prediction/
├── dataset/
│   └── heart.csv
├── models/
│   └── heart_disease_pipeline.joblib   # created after training
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── app.py
├── train_model.py
└── requirements.txt
```

## 1. Open the folder in VS Code

Open the `Heart_Disease_Prediction` folder in VS Code. Make sure `dataset/heart.csv` exists.

## 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\\Scripts\\activate.bat
```

## 3. Install packages

```bash
python -m pip install -r requirements.txt
```

## 4. Train the machine-learning model

```bash
python train_model.py
```

The script prints evaluation metrics and creates:

`models/heart_disease_pipeline.joblib`

## 5. Start the web application

```bash
uvicorn app:app --reload
```

Open this address in your browser:

`http://127.0.0.1:8000`

API documentation is available at:

`http://127.0.0.1:8000/docs`

## Dataset notes

The supplied CSV has 920 rows and 16 columns. The model uses the available health-related columns and excludes `id` and `dataset` because those are identifiers/source labels, not patient measurements. The original `num` target is converted to binary to match the supplied project approach: `0` stays 0 and values greater than `0` become 1.

The pipeline imputes missing values and performs encoding/scaling inside the training split to avoid fitting preprocessing on the test data.

## Important limitation

This is an academic demonstration, not a clinically validated medical device. Its output is not a diagnosis and should not guide treatment or delay medical care. The probability shown is a model estimate, not an individual's confirmed medical risk.

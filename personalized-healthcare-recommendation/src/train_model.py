from pathlib import Path
from model_utils import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]
metrics = train_and_save(
    PROJECT_ROOT / "data" / "raw" / "Cleaned_Dataset.csv",
    PROJECT_ROOT / "models" / "best_model.pkl",
    PROJECT_ROOT / "models" / "disease_encoder.pkl",
)
print(metrics)

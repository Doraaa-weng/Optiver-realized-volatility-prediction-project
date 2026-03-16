"""
Run full pipeline locally: create synthetic data (if no parquet) -> train -> inference.
On Kaggle Notebook: add competition data and run only train + inference (no synthetic data).
"""
from pathlib import Path
import sys

from utils import get_data_dir

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    data_dir = get_data_dir()
    artifact_root = PROJECT_DIR / "artifacts"
    p1 = data_dir / "synthetic_data" / "book_train.parquet"
    p2 = data_dir / "book_train.parquet"
    has_parquet = (p1.exists() and p1.is_file()) or (p2.exists() and p2.is_file())

    if not has_parquet:
        print("No book/trade parquet found. Creating synthetic data (subset)...")
        from create_synthetic_data import main as create_data
        create_data(max_train_keys=8000)
        print()
    else:
        print("Using existing parquet data.")

    print("Training...")
    from train import run_train
    run_train(data_dir=data_dir, output_dir=artifact_root, save_model=True)
    print()

    print("Inference -> artifacts/submissions/submission.csv")
    from inference import run_inference
    run_inference(
        data_dir=data_dir,
        model_dir=artifact_root,
        output_path=artifact_root / "submissions" / "submission.csv",
    )
    print("Done. Check artifacts/submissions/submission.csv.")


if __name__ == "__main__":
    main()
    sys.exit(0)

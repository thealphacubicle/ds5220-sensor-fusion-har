import pandas as pd
from pathlib import Path
import urllib.request
import zipfile

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ZIP_PATH = RAW_DIR / "HAR_wrapper.zip"
WRAPPER_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATASET_DIR = RAW_DIR / "UCI HAR Dataset"
NESTED_ZIP = RAW_DIR / "UCI HAR Dataset.zip"


# === Activity Mapping ===
def load_activity_map():
    df = pd.read_csv(DATASET_DIR / "activity_labels.txt", sep=r'\s+', header=None, names=["id", "label"])
    return dict(zip(df["id"], df["label"]))


# === Feature Names (fixed for duplicates) ===
def load_feature_names():
    df = pd.read_csv(DATASET_DIR / "features.txt", sep=r'\s+', header=None, names=["index", "feature"])
    names = df["feature"].tolist()

    # Fix duplicates by adding suffixes
    seen = {}
    unique_names = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}__{seen[name]}")
    return unique_names


# === Downloader ===
def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists():
        print(f"[✓] Dataset already exists at {DATASET_DIR}")
        return

    print(f"[⏳] Downloading dataset from {WRAPPER_URL}")
    urllib.request.urlretrieve(WRAPPER_URL, ZIP_PATH)
    print(f"[✓] Downloaded to {ZIP_PATH}")

    print("[📦] Extracting outer zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(RAW_DIR)

    if NESTED_ZIP.exists():
        print("[📦] Extracting inner zip...")
        with zipfile.ZipFile(NESTED_ZIP, 'r') as zf:
            zf.extractall(RAW_DIR)
        NESTED_ZIP.unlink()

    ZIP_PATH.unlink()
    print(f"[✓] Dataset ready at {DATASET_DIR}")


# === Preprocessing ===
def load_split(split="train"):
    feature_names = load_feature_names()
    activity_map = load_activity_map()

    X_path = DATASET_DIR / f"{split}/X_{split}.txt"
    y_path = DATASET_DIR / f"{split}/y_{split}.txt"
    subject_path = DATASET_DIR / f"{split}/subject_{split}.txt"

    X = pd.read_csv(X_path, sep=r'\s+', header=None, names=feature_names)
    y = pd.read_csv(y_path, header=None, names=["activity_id"])
    subject = pd.read_csv(subject_path, header=None, names=["subject"])

    df = pd.concat([X, y, subject], axis=1)
    df["activity"] = df["activity_id"].map(activity_map)
    return df


def preprocess_and_save():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_split("train")
    df_test = load_split("test")

    # Ensure no subject leakage between train and test sets
    overlap = set(df_train["subject"]).intersection(set(df_test["subject"]))

    if overlap:
        print(f"[!] Found overlapping subject ids: {sorted(overlap)}. Removing them.")
        df_train = df_train[~df_train["subject"].isin(overlap)]
        df_test = df_test[~df_test["subject"].isin(overlap)]

    else:
        print("[✓] No overlapping subjects found.")

    print(f"[→] Final train shape: {df_train.shape}")
    print(f"[→] Final test shape: {df_test.shape}")

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print("[✓] Saved to data/processed/: train.parquet, test.parquet")


# === Entry Point ===
if __name__ == "__main__":
    download_and_extract()
    preprocess_and_save()

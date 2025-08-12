# 📁 Data Directory

This folder contains all datasets used in the **Sensor Fusion for Human Activity Recognition** project.

## 📂 Folder Structure

* `raw/`: Contains the original UCI HAR dataset downloaded from the UCI Machine Learning Repository
* `processed/`: Contains cleaned `.parquet` files ready for model training and evaluation
* `README.md`: This file — explains how the data pipeline works

**Note:** Both `raw/` and `processed/` are excluded from version control using `.gitignore` to keep the repository lightweight.

## ⚙️ How to Populate This Folder

To prepare the dataset locally, run the preprocessing script:

```bash
python src/utils/preprocess.py
```

This script will:

1. **Download** the UCI HAR dataset from the official source if it hasn't been downloaded already.
2. **Extract** the outer and inner ZIP files into `data/raw/`
3. **Load and merge**:

   * Feature data (`X_train.txt` / `X_test.txt`)
   * Activity labels (`y_train.txt` / `y_test.txt`)
   * Subject IDs (`subject_train.txt` / `subject_test.txt`)
4. **Map** `activity_id` (1–6) to human-readable labels like `WALKING`, `STANDING`, etc.
5. **Assign column names** to the 561 features using `features.txt`. Any duplicate feature names are de-duplicated with suffixes.
6. **Save** the final result as:

   * `data/processed/train.parquet`
   * `data/processed/test.parquet`

---

## 📝 Output Format

Each `.parquet` file includes:

* Five hundred sixty-one extracted sensor features
* `activity_id`: numerical label
* `activity`: corresponding activity name
* `subject`: subject identifier (1–30)

---

## ✅ Output File Shapes

| File          | Rows | Columns |
|---------------|------|---------|
| train.parquet | 7352 | 564     |
| test.parquet  | 2947 | 564     |


- **Note**: To view the actual data frames with the respective information, run `src/experiments/quickstart.ipynb`.
---

If you run into issues, make sure:

* You have an internet connection (for the first-time download)
* Your Python environment includes `pandas`, `pyarrow`, and `zipfile`

# 🚨 Sensor Anomaly Detection

Predicting anomalies from sensor readings using classical and advanced machine learning models on a highly imbalanced dataset.

---

## 📌 Problem Statement

Given time-series sensor readings (`X1` to `X5`), predict whether a reading is anomalous (`target = 1`) or normal (`target = 0`). The dataset is extremely imbalanced — only **0.07%** of readings are anomalies.

---

## 📁 Project Structure

```
├── notebook.ipynb        # Main Jupyter Notebook with all code
├── submission.csv        # Final predictions for competition submission
└── README.md             # Project documentation
```

---

## 📊 Dataset

| File | Description |
|---|---|
| `train.parquet` | Training data with sensor readings and target labels |
| `test.parquet` | Test data (no labels) for prediction |
| `sample_submission.parquet` | Submission format reference |

**Features:** `X1, X2, X3, X4, X5` (sensor readings)  
**Target:** `0` = Normal, `1` = Anomaly  
**Class ratio:** ~1382:1 (normal:anomaly)

---

## 🔧 Methodology

### 1. Data Exploration & Preprocessing
- Checked missing values and filled with **column median**
- Removed outliers using **IQR method**
- **Correlation heatmap** and variance analysis
- **Feature engineering**: interaction terms (`X1*X2`, `X3*X4`) and polynomial features (`X1²`)
- Scaled features using **StandardScaler**

### 2. Models Used

#### Classical Models
| Model | Key Setting |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| SVM (LinearSVC) | `class_weight='balanced'` |
| Decision Tree | `max_depth=6`, `class_weight='balanced'` |
| KNN | `n_neighbors=5` |

#### Advanced Models
| Model | Key Setting |
|---|---|
| Random Forest | `class_weight='balanced'`, `n_estimators=20` |
| XGBoost | `scale_pos_weight=1382`, `eval_metric='aucpr'` |
| LightGBM | `class_weight='balanced'` |
| CatBoost | `auto_class_weights='Balanced'` |
| Neural Network (MLP) | `hidden_layer_sizes=(64, 32)` |

### 3. Handling Class Imbalance
- `scale_pos_weight` in XGBoost
- `class_weight='balanced'` in sklearn models
- `auto_class_weights='Balanced'` in CatBoost
- **Threshold tuning** on validation set to maximize F1 score

### 4. Model Evaluation
- Metrics: **Accuracy, Precision, Recall, F1 Score**
- **Confusion matrices** for all models
- **5-Fold Stratified Cross-Validation**
- **ROC curves** with AUC scores
- **Residual/error analysis** (FP/FN breakdown)
- Model comparison summary table sorted by F1

---

## 📈 Results

| Model | F1 Score |
|---|---|
| XGBoost | Best |
| LightGBM | - |
| CatBoost | - |
| Random Forest | - |
| Neural Network | - |
| Logistic Regression | - |
| SVM | - |
| Decision Tree | - |
| KNN | - |

> F1 Score is the primary metric due to severe class imbalance.

---

## 🚀 How to Run

1. Open the notebook on **Kaggle**
2. Attach the competition dataset
3. Run all cells in order (`Run All`)
4. `submission.csv` will be generated in the output tab
5. Submit via the competition page

---

## 📦 Dependencies

```python
pandas, numpy, matplotlib, seaborn
scikit-learn
xgboost
lightgbm
catboost
```

All pre-installed in the Kaggle Python environment.

---

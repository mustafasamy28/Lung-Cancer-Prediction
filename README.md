# 🫁 Lung Cancer Prediction

This project applies supervised machine learning techniques to predict the likelihood of **lung cancer** based on various behavioral, physiological, and lifestyle features. The project demonstrates a complete data science pipeline, including EDA, preprocessing, class balancing, model selection, and hyperparameter tuning.

## 📊 Dataset Description

The dataset includes **309 records**, each representing a person with features relevant to lung cancer risk factors. The target column is `LUNG_CANCER`, a binary classification label (YES/NO).

### 🔹 Features

| Column                  | Description                                |
|-------------------------|--------------------------------------------|
| `GENDER`                | Gender of the individual                   |
| `AGE`                   | Age in years                               |
| `SMOKING`               | Smoking status                             |
| `YELLOW_FINGERS`        | Yellow-stained fingers                     |
| `ANXIETY`               | Suffers from anxiety                       |
| `PEER_PRESSURE`         | Peer pressure presence                     |
| `CHRONIC DISEASE`       | Has chronic disease                        |
| `FATIGUE`               | Experience of fatigue                      |
| `ALLERGY`               | Allergy status                             |
| `WHEEZING`              | Wheezing symptom                          |
| `ALCOHOL CONSUMING`     | Alcohol consumption                        |
| `COUGHING`              | Cough frequency                            |
| `SHORTNESS OF BREATH`   | Breathing difficulty                       |
| `SWALLOWING DIFFICULTY` | Trouble swallowing                         |
| `CHEST PAIN`            | Chest pain presence                        |
| `LUNG_CANCER`           | Target label (YES/NO)                      |

## ⚙️ Project Workflow

### ✅ 1. Data Cleaning
- Removed duplicates
- Handled categorical features using `LabelEncoder`
- Verified absence of missing values

### ✅ 2. Exploratory Data Analysis (EDA)
- Visualized class imbalance
- Analyzed feature distributions and pairwise correlations

### ✅ 3. Outlier Detection
- Used the IQR method to remove outliers from numerical columns

### ✅ 4. Class Balancing
- Applied **SMOTE** to balance the target class distribution (YES vs NO)

### ✅ 5. Baseline Model
- Trained a **Logistic Regression** model to establish performance baseline

### ✅ 6. Model Selection & Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross-validation to tune the following models:

| Model            | Best Parameters                                      | Best CV Accuracy |
|------------------|------------------------------------------------------|------------------|
| Logistic         | `C=10`, `solver='lbfgs'`                             | **0.9218**       |
| Random Forest    | `n_estimators=200`, `max_depth=None`                | 0.9216           |
| XGBoost          | `learning_rate=0.1`, `max_depth=3`, `n_estimators=100` | **0.9218**    |

> ✅ **Best Model Selected:** Logistic Regression

## 🧪 Final Model Evaluation

```
              precision    recall  f1-score   support

           0       0.80      1.00      0.89         8
           1       1.00      0.96      0.98        47

    accuracy                           0.96        55
   macro avg       0.90      0.98      0.93        55
weighted avg       0.97      0.96      0.97        55
```

## 📈 Skills Demonstrated

- Data preprocessing & cleaning
- Exploratory data analysis (EDA)
- Outlier detection (IQR method)
- Class imbalance handling (SMOTE)
- Model comparison & tuning with GridSearchCV
- Evaluation using classification metrics


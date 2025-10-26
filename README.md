# Obesity Level Prediction Using Machine Learning

**Ashutosh Panda** (MT2025025)  
**Mihir Bindal** (MT2025072)  
[Ashutosh's GitHub](https://github.com/<your-org-or-user>/<repo>)  
[Mihir's GitHub](https://github.com/MihirBindal/ML-Project)  

## Abstract

This report documents the process of creating a supervised model that predicts obesity level from lifestyle and biometric features. It covers exploratory data analysis, data processing, model baselines, XGBoost tuning with cross-validation, and final results suitable for Kaggle submission.

## Introduction

The dataset consists of the estimation of obesity levels in people aged between 14 and 61 with diverse eating habits and physical conditions, represented by 17 attributes and 15,533 records. The goal is to create a model that can predict the weight category out of 7 possible classes ranging from `Insufficient_Weight` to `Obesity_Type_III`.

## Data Processing

### Exploratory Data Analysis

We used `ydata_profiling` for initial analysis, which provided insights into missing values, duplicate rows, correlations, and feature distributions.

Key observations:
- **Weight Category Distribution**: The weight category we aim to predict is not equally distributed across its 7 classes. `Obesity_Type_III` has a lot more entries, which requires attention during dataset balancing.
  ![Weight Category Distribution](y_freq.png)

- **Age Distribution**: The age distribution forms an off-center bell curve with a bias towards the 20-30 year age group, suggesting the dataset has more people in their 20s.
  ![Age Distribution](age.png)

- **Height Distribution**: The height distribution is a bell curve with most people having a height around 1.7 meters, which doesn't seem to be a major differentiator.
  ![Height Distribution](height.png)

- **Weight Distribution**: The weight distribution is more spread out, with a wide range of values. The high standard deviation suggests significant variability.
  ![Weight Distribution](weight.png)

- **Weight Category and Weight Relationship**: There is a clear correlation between the weight and weight category.
  ![Weight Category vs Weight](weighty.png)

- **Gender vs Weight Category**: A heatmap reveals that female weight categories skew more towards the extremes, while male categories are more centered.
  ![Gender vs Weight Category](genderh.png)

- **FAVC (Frequent Consumption of High Caloric Food)**: The FAVC column is highly imbalanced, with 91% of rows marked as `yes`. However, it helps distinguish between obese and non-obese individuals.
  ![FAVC Distribution](favc.png)

- **Family History of Obesity**: The family history feature correlates strongly with weight category. Those without a family history are more likely to fall in the `Insufficient_Weight`, `Normal`, or `Overweight` categories.
  ![Family History of Obesity](fh.png)

### Target and Feature Engineering

- **WeightCategory Encoding**: We mapped the `WeightCategory` to ordinal labels `{0, 1, ..., 6}` based on the severity of obesity.
- **HealthyEatingScore**: We created a PCA-derived score using `FCVC`, `CH2O`, and a signed mapping of `FAVC` (yes → negative weight). The first principal component coefficients were approximately `(0.659, 0.696, -0.285)`.

### Encoding and Scaling

- **Nominal Features**: Features like `Gender`, `MTRANS`, `FAVC`, `SCC`, and `family_history_with_overweight` were one-hot encoded.
- **Ordinal Features**: `CAEC` and `CALC` were encoded with the levels `{no, Sometimes, Frequently, Always}`.
- **Numerical Features**: Features like `Age`, `Height`, `Weight`, and `HealthyEatingScore_PCA` were standardized.

### Train-Test Split

We used a 75%/25% train-test split, resulting in `(11649, 20)` training features and `(3884, 20)` test features.

## Models and Evaluation

### Baselines

We trained the following models on the dataset:
- Decision Tree
- AdaBoost
- Random Forest
- XGBoost (default parameters)

**Test accuracies**:
- Decision Tree: 0.849
- AdaBoost: 0.697
- Random Forest: 0.895
- XGBoost (default): 0.903

### Hyperparameter Tuning (GridSearchCV)

We tuned the XGBoost model using a reduced grid to optimize for runtime and reproducibility. The best cross-validation accuracy was approximately `0.903`, and the final test accuracy reached `0.911`.

**Best Parameters**:
- `n_estimators`: 1032
- `max_depth`: 7
- `min_child_weight`: 5
- `max_leaves`: 31
- `learning_rate`: 0.0164
- `gamma`: 1.0935
- `reg_lambda`: 2.25
- `reg_alpha`: 0.0259
- `subsample`: 0.7105
- `colsample_bytree`: 0.572

## Results and Discussion

### Model Comparison

| Model                | Test Accuracy |
|----------------------|---------------|
| AdaBoost             | 0.697         |
| Decision Tree        | 0.849         |
| Random Forest        | 0.895         |
| XGBoost (default)    | 0.903         |
| XGBoost (tuned)      | 0.911         |

### Confusion Matrix

We included a confusion matrix for the tuned model to visualize per-class performance and confusions between adjacent obesity levels.

## Reproducibility and Submission

The final model was trained using the processed training set and used to predict test labels for Kaggle submission.

## Conclusion

By engineering nutrition features via PCA, encoding appropriately, and applying gradient boosting with careful regularization, we achieved the best performance among all tested models. Future work could explore ordinal-aware objectives or cost-sensitive training to reduce confusion between adjacent severity levels.

## Appendix

- **Environment**: Python 3.12, scikit-learn, XGBoost, ydata-profiling.
- **GitHub**: [Repository Link](<insert-your-repository-link-here>)

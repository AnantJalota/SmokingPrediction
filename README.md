## Problem Statement

The objective of this project is to build a binary classification system to predict whether an individual is a smoker or a non-smoker using physiological and biological signal data. The target variable is smoking status, and the task involves learning complex, non-linear relationships between health indicators and smoking behavior.

## Dataset Description

The dataset contains **55,287 instances** and **27 columns**. The target variable is **\'smoking\'**, where 1 indicates a smoker and 0 indicates a non-smoker. The features include a mix of numerical, binary, and categorical attributes representing physiological information and medical biomarkers.

- Gender (M/F)

- Age (5-year intervals)

- Basic Body measurements: Height (cm), Weight (kg), Waist (cm)

- Vision (left/right), Hearing (left/right)

- Blood pressure (systolic, diastolic)

- Fasting blood sugar

- Heart Health: Cholesterol, Triglycerides, HDL, LDL

- Hemoglobin

- Kidney Health: Urine protein, Serum creatinine

- Liver enzymes (AST, ALT, GTP)

- Oral examination, Dental caries, Tartar

The dataset does not contain missing values. Preprocessing steps include outlier handling for extreme values, encoding of categorical variables, feature scaling for distance-based models, and handling class imbalance using class weights or model-specific imbalance parameters. Approximately, **64% of the samples are non-smokers and 36% are smokers**.

## Metrics And Model Performance

| **ML Model Name**        | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|--------------------------|--------------|---------|---------------|------------|--------|---------|
| Logistic Regression      | 0.75         | 0.73    | 0.65          | 0.69       | 0.67   | 0.46    |
| Decision Tree            | 0.74         | 0.76    | 0.60          | 0.85       | 0.71   | 0.50    |
| KNN                      | 0.72         | 0.76    | 0.57          | 0.92       | 0.71   | 0.51    |
| Naive Bayes              | 0.71         | 0.76    | 0.57          | 0.93       | 0.70   | 0.51    |
| Random Forest (Ensemble) | 0.78         | 0.77    | 0.70          | 0.73       | 0.71   | 0.54    |
| XGBoost (Ensemble)       | 0.79         | 0.78    | 0.71          | 0.74       | 0.73   | 0.56    |

## Observations

Below are the Model performance observations, where I have also added what was done and how has that impacted the metric collected and added above.

| **ML Model Name**        | **Observation about model performance**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Since the dataset is not linearly separable, a simple Logistic Regression does not perform that well. Using **2^nd^ degree** polynomial and **L2** Regularization gives a balanced result. And, as per the metrics, it gives a balanced and a good precision-recall tradeoff.                                                                                                                                                                                                                                             |
| Decision Tree            | Decision tree without tuning severely overfits, but upon tuning it gives a slightly improved metric in general. Since there is an imbalance, I have given higher class weight slightly to the minority class. This generalization improves the recall significantly at the cost of precision as the model, to balance, predicts "smoker" more often.                                                                                                                                                                      |
| KNN                      | K-Nearest Neighbors performs surprisingly competitively with Decision Tree. The model generalizes better after feature scaling and tuning the number of neighbors. Recall improves for smokers with appropriate neighbor size selection. I have also tweaked the probability threshold in this model to improve its smoker prediction.                                                                                                                                                                                    |
| Naive Bayes              | Naïve Bayes is a generative model that I have used as it is without many changes in the parameters. The model performance as per the metrics performance is well compared to its peer models and considering the data is not straightforward. I have used GaussianNB model here to classify. But, like other models, Naïve Bayes model (because of imbalance adjustment) also has slightly higher impetus on the smoker class and therefore we see a very high recall for the "smoker" class but a not so good precision. |
| Random Forest (Ensemble) | Random Forest is an ensemble model. The key difference here is that I have used feature interactions to get a clearer combination which could cleanly separate the data when passed to a tree. Doing so leads to better precision and a stable recall. With the same imbalance as in decision tree but with more feature interactions along with more estimators the performance improves.                                                                                                                                |
| XGBoost (Ensemble)       | XGBoost is also an ensemble model. We implement a similar strategy like Random Forest along with using regularization parameters. These tweaks positively impact the model performance considering better separability between the 2 classes as evident by a significant increase in the precision at a cost on the recall. Overall, the model performance is better balanced with significant improvements compared to other models.                                                                                     |

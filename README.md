# Credit Card Fraud Detection

## Overview
This project applies machine learning to detect fraudulent credit card transactions from an imbalanced dataset.  
We train **5 classification models** (Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost) and compare their performance.  

The workflow covers:
- Data exploration and visualization  
- Feature scaling  
- Handling class imbalance with **Random Under-Sampling**  
- Model training & evaluation  
- Performance comparison (Accuracy, Precision, Recall, F1-Score, F1 comparison chart)

## Dataset
- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- 284,807 transactions, with only 492 frauds (~0.17%).  
- Features `V1–V28` (PCA transformed), plus `Time` and `Amount`.  
- Target column: `Class` (1 = Fraud, 0 = Legit).  

## Results
- **XGBoost** achieved the best balance between Precision & Recall (highest F1-score).  
- **Random Forest** also performed strongly.  

![Model F1-Score Comparison](model_comparison.png)

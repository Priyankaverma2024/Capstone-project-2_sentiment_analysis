
# Product Review Sentiment Analysis
This project perform multi-class sentiment classificatio (Positive,Negative,Neutral)
# Project Overview

# Objective
To build an end to end sentiment analysis pipeline using traditional ML Models like
(Naive Bayes, SVM, MLP, XGBoost) and LSTM, and deploy the best model with a Streamlit interface.

# Dataset 
train.csv
test.csv
test_hidden.csv
- **Text Data**: Product reviews (`review.text`)
- **Target**: Sentiment (`positive`, `neutral`, `negative`)

## 🧪 Features & Workflow

1. **Exploratory Data Analysis (EDA)**
2. **Text Preprocessing**:
   - Cleaning, removing stopwords
   - Lemmatization
3. **Feature Engineering**:
   - TF-IDF Vectorization
   - Target Encoding
4. **Model Training**:
   - Multinomial Naive Bayes
   - Support Vector Machine (SVM)
   - MLP Classifier
   - XGBoost
   - LSTM (Keras)
5. **Handling Imbalance**:
   - SMOTE oversampling
6. **Evaluation**:
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score
7. **Hyperparameter Tuning**:
   - `RandomizedSearchCV` for all models
8. **Final Prediction on Hidden Data**
9. **Streamlit Web App**

---

## 🚀 Streamlit Web App

### Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py

# Folder Structuer
sentiment_app/
├── app.py
├── requirements.txt
├── README.md
├── tfidf_vectorizer.pkl
├── XGBoost_model.pkl
├── label_encoder.pkl
├── train_data.csv
├── test_data.csv
└── test_data_hidden.csv


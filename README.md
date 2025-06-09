# Fake News Classification using Bag of N-Grams

This project focuses on classifying news articles as **fake** or **real** using Natural Language Processing (NLP) techniques and a **Bag of N-Grams** model. It demonstrates a pipeline built with Python, scikit-learn, and spaCy.

---

## Project Structure

- **`NLP_BagOfNGrams.ipynb`**: Main Jupyter Notebook that handles loading the data, preprocessing using spaCy, feature extraction with CountVectorizer (Bag of N-Grams), and classification using Naive Bayes.
- **`Fake_Real_Data.csv`**: Dataset containing labeled news articles for binary classification (fake vs real).

---

## Features & Techniques

- **Text Preprocessing**:
  - Lemmatization
  - Stopword and punctuation removal
  - spaCy's `en_core_web_sm` model
- **Vectorization**:
  - `CountVectorizer` with n-gram tokenization
- **Modeling**:
  - `MultinomialNB` (Naive Bayes Classifier)
- **Pipeline**:
  - Sklearn pipeline integrates preprocessing, vectorization, and modeling
- **Evaluation**:
  - Classification report with precision, recall, and F1-score

---

## Sample Results
The model achieves strong performance using basic preprocessing and N-Gram features. Metrics like accuracy and F1-score are printed in the final cells of the notebook using classification_report().

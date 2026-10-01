# Credit Card Fraud Detection Using Autoencoders

## Overview

This project uses a **Deep Autoencoder** to detect fraudulent credit card transactions through anomaly detection. The model learns the patterns of legitimate transactions and identifies transactions with unusually high reconstruction error as potential fraud.

## Dataset

The project uses the **Credit Card Fraud Detection dataset**:

* 284,807 total transactions
* 284,315 legitimate transactions
* 492 fraudulent transactions
* 30 features
* `Class = 0` → Legitimate
* `Class = 1` → Fraud

The dataset is highly imbalanced, making anomaly detection a useful approach.

## Approach

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train Autoencoder on Normal Transactions
   ↓
Reconstruct Transactions
   ↓
Calculate Reconstruction Error
   ↓
Apply Threshold
   ↓
Fraud / Normal
```

The autoencoder learns to reconstruct normal transactions accurately. Transactions with a reconstruction error above a selected threshold are classified as fraudulent.

## Technologies

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Evaluation

The model is evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall Curve

Precision and recall are particularly important because of the extreme class imbalance.


## Project Structure

```text
├── Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras-master/
│   ├── fraud_detection.ipynb   # Main Jupyter notebook with model training & analysis
│   ├── model.h5                # Pre-trained Keras model weights
│   ├── logs/                   # Training logs
│   └── LICENSE                 # License file
└── README.md                   # Project documentation
```

## Quick Start

1. Install required dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Launch the notebook:
   ```bash
   jupyter notebook Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras-master/fraud_detection.ipynb
   ```

## Purpose

This project demonstrates how **autoencoders can be used for anomaly detection** in highly imbalanced fraud detection problems.

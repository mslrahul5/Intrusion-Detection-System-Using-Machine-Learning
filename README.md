# Intrusion-Detection-System-Using-Machine-Learning


A machine learning–based Network Intrusion Detection System (IDS) that classifies network traffic as **normal** or **attack**, benchmarking five ML algorithms on the industry-standard **NSL-KDD** dataset.

## Overview

Traditional signature-based IDS tools (e.g., Snort, Suricata) rely on predefined attack rules and struggle to detect new or evolving threats (zero-day attacks). This project applies machine learning to detect intrusions by learning patterns from network traffic data, enabling detection of both known and previously unseen attack types.

## Dataset

- **NSL-KDD** — an improved version of the classic KDD Cup 1999 dataset, widely used as a benchmark for IDS research.
- 41 network traffic features (protocol type, service, flag, byte counts, connection statistics, etc.)
- Attack categories: **DoS**, **Probe**, **R2L** (Remote-to-Local), **U2R** (User-to-Root) — collapsed into a binary `normal` vs `attack` label for this project.
- Files: `KDDTrain+.txt` (125,973 records), `KDDTest+.txt` (22,544 records)

## Methodology

1. **Data Collection** – Load NSL-KDD train/test sets
2. **Data Preprocessing** – Encode categorical features (protocol_type, service, flag), binarize attack labels
3. **Feature Scaling** – Standardize features with `StandardScaler`
4. **Model Training** – Train 5 classifiers on the processed data
5. **Evaluation** – Compare models on accuracy, precision/recall, training time, and testing time

## Models Compared

| Model | Description |
|---|---|
| Decision Tree | Simple, interpretable baseline |
| Random Forest | Ensemble of decision trees |
| SVM | Kernel-based classifier (RBF) |
| Gradient Boosting | Sequential boosted ensemble |
| ANN (MLP) | Multi-layer perceptron neural network |

## Results

| Model | Train Accuracy | Test Accuracy | Training Time | Testing Time |
|---|---|---|---|---|
| Decision Tree | 99.99% | 78.94% | 0.75s | 0.004s |
| Random Forest | 99.99% | 77.31% | 5.81s | 0.11s |
| SVM | 99.26% | 78.20% | 26.47s | 5.69s |
| Gradient Boosting | 99.62% | 80.64% | 16.83s | 0.03s |
| ANN (MLP) | 99.78% | 78.06% | 23.73s | 0.01s |

**Gradient Boosting** achieved the best testing accuracy, while **Decision Tree** offered the fastest training and inference — a good trade-off for real-time deployment.

*(See `/outputs` for the full comparison charts and confusion matrix.)*

## Tech Stack

- **Python**
- **Scikit-learn** – model training & evaluation
- **Pandas / NumPy** – data processing
- **Matplotlib / Seaborn** – visualization

## Project Structure

├── ids_ml_pipeline.py          # Main script: preprocessing → training → evaluation
├── KDDTrain+.txt                # Training dataset
├── KDDTest+.txt                 # Testing dataset
├── model_comparison_results.csv # Final metrics table
└── outputs/
    ├── testing_accuracy.png
    ├── training_accuracy.png
    ├── training_time.png
    ├── testing_time.png
    └── confusion_matrix_best_model.png


## How to Run

bash
pip install pandas numpy scikit-learn matplotlib seaborn
python ids_ml_pipeline.py

## Future Improvements

- Multi-class classification (DoS / Probe / R2L / U2R) instead of binary
- Hyperparameter tuning (GridSearchCV) for each model
- Feature selection to reduce dimensionality and improve real-time performance
- Deployment as a real-time traffic monitoring API


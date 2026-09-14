"""
Intrusion Detection System using Machine Learning
Dataset : NSL-KDD (KDDTrain+.txt / KDDTest+.txt)
 
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

 
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

print("Loading NSL-KDD train/test files ...")
train_df = pd.read_csv("KDDTrain+.txt", names=COLUMN_NAMES)
test_df = pd.read_csv("KDDTest+.txt", names=COLUMN_NAMES)

# Drop the "difficulty" column — not a feature
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"], inplace=True)

print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

 
 
train_df["label"] = train_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
test_df["label"] = test_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

# Encode categorical columns (protocol_type, service, flag)
categorical_cols = ["protocol_type", "service", "flag"]
combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]])
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(combined[col])
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    encoders[col] = le

# Encode the label column
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])   # normal=0/1
test_df["label"] = label_encoder.transform(test_df["label"])

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_train

    start_test = time.time()
    y_pred_test = model.predict(X_test_scaled)
    testing_time = time.time() - start_test

    y_pred_train = model.predict(X_train_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time": training_time,
        "testing_time": testing_time,
    }

    print(f"  Training Accuracy : {train_acc:.4f}")
    print(f"  Testing Accuracy  : {test_acc:.4f}")
    print(f"  Training Time     : {training_time:.2f}s")
    print(f"  Testing Time      : {testing_time:.4f}s")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
 
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison_results.csv")
print("\n=== Final Comparison Table ===")
print(results_df)

def bar_plot(column, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df[column], palette="viridis")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

bar_plot("test_accuracy", "Testing Accuracy Comparison", "Accuracy", "testing_accuracy.png")
bar_plot("train_accuracy", "Training Accuracy Comparison", "Accuracy", "training_accuracy.png")
bar_plot("training_time", "Training Time Comparison", "Time (seconds)", "training_time.png")
bar_plot("testing_time", "Testing Time Comparison", "Time (seconds)", "testing_time.png")

# Confusion matrix for the best model (highest test accuracy)
best_model_name = results_df["test_accuracy"].astype(float).idxmax()
print(f"\nBest model by testing accuracy: {best_model_name}")
best_model = models[best_model_name]
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=150)
plt.close()

print("\nDone. Saved: model_comparison_results.csv, testing_accuracy.png, "
      "training_accuracy.png, training_time.png, testing_time.png, "
      "confusion_matrix_best_model.png")

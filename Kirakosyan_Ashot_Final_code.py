import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from IPython.display import display  # For displaying tables in Jupyter
from IPython import get_ipython
import sys
import io
import os

# Only modify stdout if not running in a Jupyter notebook environment
if not "ipykernel" in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set TensorFlow logging level to avoid unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow log messages

# Suppress warnings that may arise
import warnings
warnings.filterwarnings('ignore')

# Continue with your code...



# Fetch dataset
breast_cancer = fetch_ucirepo(id=14)

# Create DataFrame for features and targets
X = pd.DataFrame(breast_cancer.data.features)
y = breast_cancer.data.targets  # We will convert these string labels to numeric

# One-Hot Encoding for categorical variables in the features
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=encoder.get_feature_names_out(X.columns))

# Convert target labels ('no-recurrence-events', 'recurrence-events') to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y).ravel()   # This will convert 'no-recurrence-events' -> 0 and 'recurrence-events' -> 1

# Convert the numeric target labels to categorical (one-hot encoded) format for GRU
y_categorical = to_categorical(y_encoded)

# Reshape the features for the GRU model (samples, time steps, features)
X_reshaped = X_encoded.values.reshape((X_encoded.shape[0], 1, X_encoded.shape[1]))

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  # True Positive
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    
    # True Positive Rate (TPR) or Sensitivity
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    # Specificity (SPC)
    SPC = TN / (TN + FP) if (TN + FP) != 0 else 0
    
    # Positive Predictive Value (PPV)
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    
    # Negative Predictive Value (NPV)
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
    
    # False Positive Rate (FPR)
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    
    # False Discovery Rate (FDR)
    FDR = FP / (FP + TP) if (FP + TP) != 0 else 0
    
    # False Negative Rate (FNR)
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0
    
    # Accuracy (ACC)
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    
    # F1 Score
    epsilon = 1e-10  # Small epsilon to avoid division by zero
    F1 = 2 * (PPV * TPR) / (PPV + TPR + epsilon)
    
    # Brier Score (BS)
    BS = brier_score_loss(y_true, y_pred)
    
    # True Skill Statistic (TSS)
    TSS = TPR + SPC - 1
    
    # Heidke Skill Score (HSS)
    HSS = (TP + TN - (FP + FN)) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    
    # Balanced Accuracy (BACC)
    BACC = (TPR + SPC) / 2 if (TPR + SPC) != 0 else 0
    
    # Balanced Subset Sensitivity (BSS)
    BSS = (TPR + SPC) / 2 if (TPR + SPC) != 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'TPR': TPR, 'SPC': SPC,
        'PPV': PPV, 'NPV': NPV, 'FPR': FPR, 'FDR': FDR, 'FNR': FNR, 'ACC': ACC,
        'F1': F1, 'BS': BS, 'TSS': TSS, 'HSS': HSS, 'BACC' : BACC, 'BSS' : BSS,
    }


# Function to perform 10-fold cross-validation and evaluate models
def cross_validate_model(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_metrics = []
    roc_auc_scores = []
    brier_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics for this fold
        metrics = calculate_metrics(y_test, y_pred)
        fold_metrics.append(metrics)
        
        # ROC AUC score
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        roc_auc_scores.append(roc_auc)
        
        # Brier Score
        brier_score = brier_score_loss(y_test, y_pred_prob)
        brier_scores.append(brier_score)
    
    # Calculate average metrics over all folds
    avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}
    avg_roc_auc = np.mean(roc_auc_scores)
    avg_brier_score = np.mean(brier_scores)
    
    return fold_metrics, avg_metrics, avg_roc_auc, avg_brier_score

# Function to train and evaluate Random Forest
def train_random_forest():
    rf = RandomForestClassifier(random_state=42)
    fold_metrics_rf, avg_metrics_rf, avg_roc_auc_rf, avg_brier_rf = cross_validate_model(rf, X_encoded.values, y_encoded)

    # Display fold-wise and average metrics in tables
    print("Random Forest Fold-wise Metrics:")
    display(pd.DataFrame(fold_metrics_rf))
    print(f"Average Random Forest Metrics:")
    display(pd.DataFrame([avg_metrics_rf]))
    print(f"Average ROC AUC: {avg_roc_auc_rf:.2f}")
    print(f"Average Brier Score: {avg_brier_rf:.2f}")

# Function to train and evaluate Decision Tree
def train_decision_tree():
    dt = DecisionTreeClassifier(random_state=42)
    fold_metrics_dt, avg_metrics_dt, avg_roc_auc_dt, avg_brier_dt = cross_validate_model(dt, X_encoded.values, y_encoded)

    # Display fold-wise and average metrics in tables
    print("Decision Tree Fold-wise Metrics:")
    display(pd.DataFrame(fold_metrics_dt))
    print(f"Average Decision Tree Metrics:")
    display(pd.DataFrame([avg_metrics_dt]))
    print(f"Average ROC AUC: {avg_roc_auc_dt:.2f}")
    print(f"Average Brier Score: {avg_brier_dt:.2f}")

# Function to train and evaluate GRU
def train_gru():
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_metrics_gru = []
    accuracy_scores_gru = []
    roc_auc_scores_gru = []
    brier_scores_gru = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_reshaped, y_encoded), 1):
        X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Convert the target labels to one-hot encoding for GRU
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)
        
        # Define the GRU model
        model_gru = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(y_train_categorical.shape[1], activation='softmax')
        ])
        
        model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Train the GRU model on the training data
        model_gru.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test, y_test_categorical), verbose=0)
        
        # Evaluate the model on the test data
        y_pred_prob_gru = model_gru.predict(X_test)
        y_pred_gru = np.argmax(y_pred_prob_gru, axis=1)
        
        # Convert y_test_categorical back to the original label format for comparison
        y_test_labels_gru = np.argmax(y_test_categorical, axis=1)
        
        # Calculate metrics for this fold
        metrics = calculate_metrics(y_test_labels_gru, y_pred_gru)
        fold_metrics_gru.append(metrics)
        
        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test_labels_gru, y_pred_gru)
        accuracy_scores_gru.append(accuracy)
        
        # ROC AUC score
        roc_auc = roc_auc_score(y_test_labels_gru, y_pred_prob_gru[:, 1])
        roc_auc_scores_gru.append(roc_auc)
        
        # Brier Score
        brier_score = brier_score_loss(y_test_labels_gru, y_pred_prob_gru[:, 1])
        brier_scores_gru.append(brier_score)
    
    # Calculate average metrics over all folds
    avg_metrics_gru = {key: np.mean([fold[key] for fold in fold_metrics_gru]) for key in fold_metrics_gru[0].keys()}
    avg_accuracy_gru = np.mean(accuracy_scores_gru)
    avg_roc_auc_gru = np.mean(roc_auc_scores_gru)
    avg_brier_score_gru = np.mean(brier_scores_gru)
    
    print(f"GRU Fold-wise Metrics:")
    display(pd.DataFrame(fold_metrics_gru))
    print(f"Average GRU Metrics:")
    display(pd.DataFrame([avg_metrics_gru]))
    print(f"Average Accuracy (GRU): {avg_accuracy_gru:.2f}")
    print(f"Average ROC AUC (GRU): {avg_roc_auc_gru:.2f}")
    print(f"Average Brier Score (GRU): {avg_brier_score_gru:.2f}")

# Call functions to train and evaluate all models
train_random_forest()
train_decision_tree()
train_gru()
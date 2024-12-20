import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform linear probing on refined embeddings.")
    parser.add_argument("--refined_npz", type=str, required=True, help="Path to .npz file with refined embeddings (embeddings, labels, patch_ids)")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the best logistic regression model (joblib)")
    args = parser.parse_args()

    data = np.load(args.refined_npz, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    patch_ids = data['patch_ids']

    unique_patches = np.arange(len(patch_ids))
    train_idx, test_idx = train_test_split(unique_patches, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_val, y_val = embeddings[val_idx], labels[val_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    unique_classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weights_dict = {cls: w for cls, w in zip(unique_classes, class_weights_arr)}

    C_values = [0.01, 0.1, 1, 10, 100]
    best_f1 = -1
    best_model = None
    best_C = None

    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs', class_weight=class_weights_dict)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = clf
            best_C = C

    print(f"Best C: {best_C}, Val F1: {best_f1:.4f}")
    y_test_pred = best_model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    joblib.dump(best_model, args.output_model)
    print(f"Best model saved at {args.output_model}")

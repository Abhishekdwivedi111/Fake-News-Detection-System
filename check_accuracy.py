"""
Comprehensive script to check model accuracy and performance metrics
Run with: python check_accuracy.py
"""

import os
import pandas as pd
import numpy as np
import scikitLearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def check_model_accuracy():
    """
    Check the accuracy of the trained model
    """
    print("=" * 70)
    print("MODEL ACCURACY CHECK")
    print("=" * 70)
    
    # Load model
    print("\n[INFO] Loading model...")
    model, vectorizer = scikitLearn.prepare_model()
    print("[SUCCESS] Model loaded!\n")
    
    # Load training data to create test set
    print("[INFO] Loading datasets for evaluation...")
    fake_df = pd.read_csv('trainDataSet/Fake.csv')
    true_df = pd.read_csv('trainDataSet/True.csv')
    
    # Label the data
    fake_df['label'] = 0  # FAKE
    true_df['label'] = 1  # REAL
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df['content'] = scikitLearn.combine_text(df)
    df['cleaned'] = df['content'].apply(scikitLearn.clean_text)
    
    # Split into train and test (same as training)
    from sklearn.model_selection import train_test_split
    X = df['cleaned']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"[INFO] Test set size: {len(X_test)} samples")
    print(f"  - FAKE samples: {(y_test == 0).sum()}")
    print(f"  - REAL samples: {(y_test == 1).sum()}\n")
    
    # Vectorize test data
    print("[INFO] Vectorizing test data...")
    X_test_vec = vectorizer.transform(X_test)
    
    # Get predictions
    print("[INFO] Generating predictions...")
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_fake = precision_score(y_test, y_pred, pos_label=0)
    precision_real = precision_score(y_test, y_pred, pos_label=1)
    recall_fake = recall_score(y_test, y_pred, pos_label=0)
    recall_real = recall_score(y_test, y_pred, pos_label=1)
    f1_fake = f1_score(y_test, y_pred, pos_label=0)
    f1_real = f1_score(y_test, y_pred, pos_label=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n" + "=" * 70)
    print("ACCURACY METRICS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("Per-Class Metrics:")
    print("-" * 70)
    print(f"\nFAKE News Detection:")
    print(f"  Precision: {precision_fake * 100:.2f}%")
    print(f"  Recall:    {recall_fake * 100:.2f}%")
    print(f"  F1-Score:  {f1_fake * 100:.2f}%")
    
    print(f"\nREAL News Detection:")
    print(f"  Precision: {precision_real * 100:.2f}%")
    print(f"  Recall:    {recall_real * 100:.2f}%")
    print(f"  F1-Score:  {f1_real * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    print(f"                Predicted")
    print(f"              FAKE    REAL")
    print(f"Actual FAKE   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"       REAL   {cm[1][0]:5d}   {cm[1][1]:5d}")
    
    # Calculate percentages
    total_fake = cm[0][0] + cm[0][1]
    total_real = cm[1][0] + cm[1][1]
    correct_fake = cm[0][0]
    correct_real = cm[1][1]
    wrong_fake_as_real = cm[0][1]
    wrong_real_as_fake = cm[1][0]
    
    print(f"\nBreakdown:")
    print(f"  FAKE news correctly identified: {correct_fake}/{total_fake} ({correct_fake/total_fake*100:.2f}%)")
    print(f"  REAL news correctly identified: {correct_real}/{total_real} ({correct_real/total_real*100:.2f}%)")
    print(f"  FAKE news misclassified as REAL: {wrong_fake_as_real} ({wrong_fake_as_real/total_fake*100:.2f}%)")
    print(f"  REAL news misclassified as FAKE: {wrong_real_as_fake} ({wrong_real_as_fake/total_real*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "-" * 70)
    print("Detailed Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    
    # Probability analysis
    print("\n" + "-" * 70)
    print("Prediction Confidence Analysis:")
    print("-" * 70)
    
    # For correctly predicted samples
    correct_mask = (y_test == y_pred)
    correct_probs = y_pred_proba[correct_mask]
    correct_real_probs = correct_probs[y_test[correct_mask] == 1, 1]  # REAL predictions
    correct_fake_probs = correct_probs[y_test[correct_mask] == 0, 0]  # FAKE predictions
    
    # For incorrectly predicted samples
    wrong_mask = (y_test != y_pred)
    if wrong_mask.sum() > 0:
        wrong_probs = y_pred_proba[wrong_mask]
        wrong_real_probs = wrong_probs[y_test[wrong_mask] == 1, 1]  # REAL misclassified as FAKE
        wrong_fake_probs = wrong_probs[y_test[wrong_mask] == 0, 0]  # FAKE misclassified as REAL
        
        print(f"\nCorrect Predictions:")
        if len(correct_real_probs) > 0:
            print(f"  REAL news - Avg confidence: {correct_real_probs.mean()*100:.2f}%")
        if len(correct_fake_probs) > 0:
            print(f"  FAKE news - Avg confidence: {correct_fake_probs.mean()*100:.2f}%")
        
        print(f"\nIncorrect Predictions:")
        if len(wrong_real_probs) > 0:
            print(f"  REAL misclassified as FAKE - Avg confidence: {wrong_real_probs.mean()*100:.2f}%")
        if len(wrong_fake_probs) > 0:
            print(f"  FAKE misclassified as REAL - Avg confidence: {wrong_fake_probs.mean()*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("Accuracy Check Complete!")
    print("=" * 70)
    
    return {
        'accuracy': accuracy,
        'precision_fake': precision_fake,
        'precision_real': precision_real,
        'recall_fake': recall_fake,
        'recall_real': recall_real,
        'f1_fake': f1_fake,
        'f1_real': f1_real,
        'confusion_matrix': cm
    }


def test_on_custom_dataset(dataset_path=None):
    """
    Test model on a custom dataset (optional)
    """
    if dataset_path is None:
        print("\n[INFO] No custom dataset provided. Skipping custom test.")
        return
    
    print("\n" + "=" * 70)
    print("TESTING ON CUSTOM DATASET")
    print("=" * 70)
    
    # This would require the custom dataset to have the same structure
    # You can extend this function based on your needs
    print("[INFO] Custom dataset testing not implemented yet.")
    print("       You can extend this function to test on your dataset.")


if __name__ == "__main__":
    # Check accuracy on test set
    metrics = check_model_accuracy()
    
    # Optional: Test on custom dataset
    # Uncomment and modify if you want to test on a different dataset
    # test_on_custom_dataset("path/to/your/dataset")

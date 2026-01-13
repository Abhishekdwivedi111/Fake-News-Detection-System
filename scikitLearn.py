import os
import pandas as pd
import re
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ---------------- TEXT CLEANING ---------------- #

def clean_text(text):
    """
    Improved text cleaning that preserves important linguistic features
    Less aggressive cleaning to maintain more context
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove Twitter handles and hashtags (but keep the text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace but keep sentence structure
    text = re.sub(r'\s+', ' ', text)
    
    # Keep numbers as they can be important (like dates, amounts)
    # Just normalize multiple spaces around numbers
    text = re.sub(r'\s+', ' ', text)
    
    # Keep more punctuation for context - only remove special chars
    # This helps preserve sentence structure and formality indicators
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
    
    # Normalize multiple punctuation marks (e.g., !!! -> !)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Final cleanup
    text = text.strip()
    
    return text


def combine_text(df):
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    return df['title'] + " " + df['text']


# ---------------- TRAIN MODEL ---------------- #

def train_and_save_model(use_test_data=False):
    """
    Train the model on training data
    If use_test_data=True, also includes testDataSet in training
    """
    print("[INFO] Loading training datasets...")

    fake_df = pd.read_csv("trainDataSet/Fake.csv")
    true_df = pd.read_csv("trainDataSet/True.csv")
    
    # Optionally include test data for better generalization
    if use_test_data:
        try:
            print("[INFO] Including test dataset for better generalization...")
            test_df = pd.read_csv("testDataSet/news.csv")
            test_df['content'] = test_df['title'].fillna('') + ' ' + test_df['text'].fillna('')
            
            test_fake = test_df[test_df['label'] == 'FAKE'][['title', 'text']].copy()
            test_true = test_df[test_df['label'] == 'REAL'][['title', 'text']].copy()
            
            # Add missing columns with default values
            for col in ['subject', 'date']:
                if col not in test_fake.columns:
                    test_fake[col] = ''
                if col not in test_true.columns:
                    test_true[col] = ''
            
            # Combine with training data
            fake_df = pd.concat([fake_df, test_fake], ignore_index=True)
            true_df = pd.concat([true_df, test_true], ignore_index=True)
            print(f"[INFO] Combined: {len(fake_df)} FAKE, {len(true_df)} REAL samples")
        except Exception as e:
            print(f"[WARNING] Could not load test data: {e}")
            print("[INFO] Continuing with training data only...")

    # Numeric labels (BEST PRACTICE)
    fake_df['label'] = 0   # Fake
    true_df['label'] = 1   # Real

    # Merge datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine + clean text
    df['content'] = combine_text(df)
    df['content'] = df['content'].apply(clean_text)

    X = df['content']
    y = df['label']

    print("[INFO] Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[INFO] Vectorizing text using TF-IDF...")
    # Improved TF-IDF with more features and better ngram range
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,  # Increased from 5000 to capture more patterns
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True,  # Apply sublinear tf scaling
        norm='l2'  # L2 normalization
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"[INFO] Feature matrix shape: {X_train_vec.shape}")

    print("[INFO] Training Random Forest model with optimized hyperparameters...")
    # Fine-tuned Random Forest with better hyperparameters
    model = RandomForestClassifier(
        n_estimators=300,  # Increased from 200 for better ensemble
        max_depth=100,  # Increased depth for more complex patterns
        min_samples_split=10,  # Increased to reduce overfitting
        min_samples_leaf=4,  # Increased to reduce overfitting
        max_features='sqrt',  # Use sqrt of features for each tree
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        bootstrap=True,
        oob_score=True,  # Out-of-bag scoring for validation
        verbose=0
    )

    print("[INFO] Fitting model (this may take a few minutes)...")
    model.fit(X_train_vec, y_train)
    
    # Print out-of-bag score
    if hasattr(model, 'oob_score_'):
        print(f"[INFO] Out-of-bag score: {model.oob_score_:.4f}")

    print("\n[INFO] Model Evaluation on Test Set:")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"[SUCCESS] Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[INFO] Confusion Matrix:")
    print(f"                Predicted")
    print(f"              FAKE    REAL")
    print(f"Actual FAKE   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"       REAL   {cm[1][0]:5d}   {cm[1][1]:5d}")
    
    # Feature importance (top 20)
    print(f"\n[INFO] Top 20 Most Important Features:")
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} ({importances[idx]:.4f})")

    print("\n[INFO] Saving model & vectorizer...")
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    print("[SUCCESS] Training completed successfully.")
    return model, vectorizer


# ---------------- LOAD MODEL ---------------- #

def prepare_model():
    if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        print("[INFO] Loaded saved model.")
    else:
        model, vectorizer = train_and_save_model()
    return model, vectorizer


# ---------------- PREDICTION ---------------- #

def predict_text(model, vectorizer, text):
    """
    Predict using raw model probabilities - trust the model's judgment
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    # Get raw prediction from model (this uses the class with highest probability)
    raw_prediction = model.predict(vector)[0]
    
    # Convert numeric prediction to label
    # Model uses: 0 = FAKE, 1 = REAL
    return "REAL" if raw_prediction == 1 else "FAKE"


def predict_content(model, vectorizer, content):
    """Alias for predict_text - used for URL content analysis"""
    return predict_text(model, vectorizer, content)


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Random Forest Fake News Detection Model")
    print("=" * 70)
    
    model, vectorizer = prepare_model()
    
    print("\n" + "-" * 70)
    print("Running Test Cases:")
    print("-" * 70)

    test_cases = [
        {
            "name": "Real News Example 1",
            "text": """
            WASHINGTON (Reuters) - The Federal Reserve announced on Wednesday that it will 
            maintain its current interest rate policy, citing stable economic growth and 
            controlled inflation. The decision was made after a two-day meeting of the 
            Federal Open Market Committee.
            """
        },
        {
            "name": "Fake News Example 1",
            "text": """
            BREAKING: Government announces free money for everyone! Just share this post 
            and send your bank details to receive $10,000 instantly. This is 100% real 
            and verified by the president himself!
            """
        },
        {
            "name": "Real News Example 2",
            "text": """
            WASHINGTON - Government officials announced on Tuesday a new skill development program 
            to improve employment opportunities for youth across the country. The program, which 
            will be implemented over the next fiscal year, focuses on digital skills and renewable 
            energy sectors. Officials said the initiative aims to create thousands of new jobs.
            """
        },
        {
            "name": "Real News Example 3",
            "text": """
            Scientists at a leading research university have published findings in a peer-reviewed 
            journal showing significant progress in renewable energy technology. The study, which 
            involved collaboration with industry partners, demonstrates improved efficiency in 
            solar panel design. Researchers say the breakthrough could reduce costs by up to 30 percent.
            """
        },
        {
            "name": "Real News Example 4",
            "text": """
            The local school board voted unanimously on Monday to approve a new curriculum that 
            emphasizes science and technology education. The decision came after months of 
            public consultation and review by education experts. The new program will begin 
            implementation in the fall semester.
            """
        },
        {
            "name": "Fake News Example 2",
            "text": """
            Government gives free laptops to all citizens starting tomorrow.
            Share your Aadhaar number and bank details to receive the benefit.
            """
        },
        {
            "name": "Fake News Example 3",
            "text": """
            SHOCKING! Doctors don't want you to know this one simple trick that cures all diseases!
            Click here now to discover the secret that big pharma doesn't want you to know!
            This will change your life forever! Share with 10 friends to unlock!
            """
        },
        {
            "name": "Fake News Example 4",
            "text": """
            BREAKING: Celebrity reveals secret that will make you rich overnight! Just forward 
            this message to 20 people and money will appear in your account. This is 100% 
            verified and guaranteed to work! Don't miss out on this amazing opportunity!
            """
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 70)
        
        # Get prediction
        prediction = predict_text(model, vectorizer, test['text'])
        
        # Get probabilities
        cleaned = clean_text(test['text'])
        vector = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(vector)[0]
        
        # Model uses: 0 = FAKE, 1 = REAL
        fake_prob = probabilities[0] * 100
        real_prob = probabilities[1] * 100
        credibility_score = int(real_prob)
        
        print(f"Prediction: {prediction}")
        print(f"Credibility Score (REAL probability): {credibility_score}%")
        print(f"FAKE probability: {fake_prob:.2f}%")
        print(f"REAL probability: {real_prob:.2f}%")
        
        # Verdict
        if prediction == "REAL" and credibility_score >= 70:
            verdict = "[OK] LIKELY REAL NEWS"
        elif prediction == "REAL" and credibility_score >= 50:
            verdict = "[?] UNCERTAIN (leaning REAL)"
        elif prediction == "FAKE" and credibility_score <= 30:
            verdict = "[X] LIKELY FAKE NEWS"
        elif prediction == "FAKE" and credibility_score <= 50:
            verdict = "[?] UNCERTAIN (leaning FAKE)"
        else:
            verdict = "[?] UNCERTAIN"
        
        print(f"Verdict: {verdict}")
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class UnifiedPhishingDetector:
    def __init__(self):
        self.text_vectorizer = None
        self.model = None
        self.feature_names = []
        
    def extract_structural_features(self, content, content_type):
        features = {}
        features['length'] = len(content)
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),?":{}|<>]', content))
        features['num_digits'] = len(re.findall(r'\d', content))
        features['num_uppercase'] = len(re.findall(r'[A-Z]', content))
        features['uppercase_ratio'] = features['num_uppercase'] / max(1, features['length'])
        
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        features['num_urls'] = len(urls)
        
        if content_type == 'email':
            features['has_urgent_keywords'] = 1 if any(word in content.lower() for word in 
                                                     ['urgent', 'immediately', 'verify', 'confirm', 'security', 'account']) else 0
        elif content_type == 'message':
            features['has_shortened_url'] = 1 if any(domain in content for domain in 
                                                   ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
        elif content_type == 'url':
            features['num_dots'] = content.count('.')
            features['num_hyphens'] = content.count('-')
            features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', content) else 0
            features['is_https'] = 1 if content.startswith('https') else 0
        
        return features
    
    def create_training_data(self):
        data = []
        
        # Phishing examples
        phishing_emails = [
            "URGENT: Your account will be suspended. Verify now: http://secure-login-verify.com",
            "Dear Customer, We detected suspicious activity. Confirm your identity: http://bank-security-update.net",
            "Security Alert: Unusual login attempt. Click here to secure: http://account-verification-center.com"
        ]
        
        legitimate_emails = [
            "Meeting reminder: Team sync at 3 PM tomorrow in Conference Room B",
            "Your monthly newsletter from Tech Company is here",
            "Password changed successfully for your account"
        ]
        
        phishing_messages = [
            "You've won $1000! Claim your prize: http://bit.ly/winprize",
            "URGENT: Your bank card is locked. Call now: 555-0123",
            "Free iPhone! Click here: http://tinyurl.com/freegift"
        ]
        
        legitimate_messages = [
            "Your Uber is arriving in 3 minutes",
            "Your verification code is 123456",
            "Your Amazon order has been delivered"
        ]
        
        phishing_urls = [
            "http://secure-login-verify-account.com",
            "https://facebook-security-confirm.net",
            "http://paypal-verification-center.com"
        ]
        
        legitimate_urls = [
            "https://google.com/search?q=hello",
            "https://github.com/user/repo",
            "https://example.com/about"
        ]
        
        # Add to dataset
        for email in phishing_emails:
            data.append({'content': email, 'content_type': 'email', 'label': 1})
        for email in legitimate_emails:
            data.append({'content': email, 'content_type': 'email', 'label': 0})
        for msg in phishing_messages:
            data.append({'content': msg, 'content_type': 'message', 'label': 1})
        for msg in legitimate_messages:
            data.append({'content': msg, 'content_type': 'message', 'label': 0})
        for url in phishing_urls:
            data.append({'content': url, 'content_type': 'url', 'label': 1})
        for url in legitimate_urls:
            data.append({'content': url, 'content_type': 'url', 'label': 0})
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        structural_features = []
        for _, row in df.iterrows():
            features = self.extract_structural_features(row['content'], row['content_type'])
            structural_features.append(features)
        
        structural_df = pd.DataFrame(structural_features)
        
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            text_features = self.text_vectorizer.fit_transform(df['content'])
        else:
            text_features = self.text_vectorizer.transform(df['content'])
        
        text_df = pd.DataFrame(text_features.toarray(), 
                             columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])
        
        final_features = pd.concat([structural_df, text_df], axis=1)
        self.feature_names = list(final_features.columns)
        
        return final_features
    
    def train(self):
        print("Creating training data...")
        df = self.create_training_data()
        print(f"Training data created: {len(df)} samples")
        
        print("Preparing features...")
        X = self.prepare_features(df)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return self
    
    def predict(self, content, content_type):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        structural_features = self.extract_structural_features(content, content_type)
        structural_df = pd.DataFrame([structural_features])
        
        text_features = self.text_vectorizer.transform([content])
        text_df = pd.DataFrame(text_features.toarray(), 
                             columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])
        
        features = pd.concat([structural_df, text_df], axis=1)
        
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_names]
        
        probability = self.model.predict_proba(features)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'is_phishing': bool(prediction),
            'content_type': content_type
        }
    
    def save(self, filename='unified_phishing_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved as {filename}")

if __name__ == "__main__":
    detector = UnifiedPhishingDetector()
    detector.train()
    detector.save()
    
    # Test the model
    test_cases = [
        ("URGENT: Verify your bank account now!", "email"),
        ("Your package has been delivered", "message"), 
        ("http://secure-login-verify.com", "url"),
        ("You won a free iPhone! Click here", "message")
    ]
    
    print("\nTesting model predictions:")
    for content, content_type in test_cases:
        result = detector.predict(content, content_type)
        print(f"Content: {content}")
        print(f"Type: {content_type}, Phishing: {result['is_phishing']}, Confidence: {result['probability']:.3f}")

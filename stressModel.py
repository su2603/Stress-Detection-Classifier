import pandas as pd
import numpy as np
import re
import string
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import shap
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"stress_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stress_model')

class StressModel:
    def __init__(self, model_dir="models"):
        """Initialize the stress model with optional model directory."""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize NLTK resources with error handling
        try:
            for r in ('stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4'):
                nltk.download(r, quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {e}")
            # Use basic fallback for stopwords if download fails
            self.stop_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it'])
            self.lemmatizer = None
        
        self.model = None
        self.tfidf = None
        self.scaler = None
        self.stress_indicators = set([
        'stress', 'anxiety', 'anxious', 'worried', 'fear', 'scared', 'terrified',
        'depressed', 'depression', 'suicide', 'suicidal', 'die', 'death', 'kill',
        'overwhelm', 'overwhelmed', 'exhausted', 'tired', 'fatigue', 'fatigued',
        'pressure', 'deadline', 'workload', 'burden', 'sad', 'unhappy',
        'hopeless', 'lonely', 'alone', 'isolated', 'miserable', 'worthless', 
        'empty', 'void', 'numb', 'pain', 'hurt', 'cry', 'tears', 'sob',
        'panic', 'attack', 'breakdown', 'sick', 'ill', 'insomnia', 'sleep',
        'anger', 'angry', 'mad', 'hate', 'despair', 'desperate', 'awful',
        'terrible', 'horrible', 'worst', 'fail', 'failure', 'shit', 'fuck', 'damn'
        ])
        self.feature_cols = [
            'sentiment', 'syntax_fk_grade', 'syntax_ari',
            'social_karma', 'social_num_comments', 'social_upvote_ratio','stress_word_count', 'negative_emotion_ratio'
        ]

    def load_data(self, file_path):
        """Load and clean data from CSV file with error handling."""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Add basic data validation
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns")
            
            # Check for and handle missing values
            initial_rows = len(df)
            df = df.dropna(subset=['text', 'label'])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing text or label")
            
            # Convert labels to integers and validate
            df['label'] = df['label'].astype(int)
            if not set(df['label'].unique()).issubset({0, 1}):
                logger.warning("Labels contain values other than 0 and 1")
            
            # Add text length as a feature
            df['text_len'] = df['text'].apply(len)
            
            # Ensure all required feature columns exist
            for col in self.feature_cols:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found, creating with zeros")
                    df[col] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def clean_text(self, text):
        """Enhanced text cleaning that preserves stress indicators."""
        if not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove square brackets and their contents
            text = re.sub(r'\[.*?\]', '', text)
            
            # IMPORTANT: We'll preserve some punctuation that might indicate stress
            # Like '...', '!!!', '???' - these can be strong emotional indicators
            # Instead of removing all punctuation, we'll normalize it
            text = re.sub(r'\.{2,}', ' ellipsis ', text)  # Replace '...' with 'ellipsis'
            text = re.sub(r'!{2,}', ' exclamation ', text)  # Replace '!!!' with 'exclamation'
            text = re.sub(r'\?{2,}', ' question ', text)  # Replace '???' with 'question'
            
            # Now remove remaining punctuation (except apostrophes for contractions)
            text = re.sub(r'[^\w\s\']', ' ', text)
            
            # Tokenize and lemmatize, but DO NOT remove stopwords that might indicate stress
            # Words like "no", "not", "never" can be crucial for sentiment
            if self.lemmatizer:
                tokens = word_tokenize(text)
                # Only remove a subset of stopwords, keeping negations and emotional words
                emotional_stopwords = {'and', 'the', 'a', 'an', 'of', 'in', 'to', 'for'}
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                         if word not in emotional_stopwords or word in self.stress_indicators]
                text = ' '.join(tokens)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""
    
    def extract_stress_features(self, texts):
        """Extract additional stress-related linguistic features from texts."""
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                features.append({'stress_word_count': 0, 'negative_emotion_ratio': 0})
                continue
                
            # Clean and tokenize
            text = text.lower()
            words = text.split()
            total_words = max(len(words), 1)  # Avoid division by zero
            
            # Count stress indicator words
            stress_words = sum(1 for word in words if word in self.stress_indicators)
            
            # Calculate negative emotion ratio
            negation_words = ['no', 'not', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "don't", "doesn't", "didn't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't"]
            negations = sum(1 for word in words if word in negation_words)
            negative_ratio = (negations + stress_words) / total_words
            
            features.append({
                'stress_word_count': stress_words,
                'negative_emotion_ratio': negative_ratio
            })
        
        return pd.DataFrame(features)
    
    def prepare_features(self, df, training=True):
        """Enhanced feature preparation with stress-specific features."""
        logger.info("Preparing features with stress-specific indicators")
        
        # Clean text
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Extract text features
        X_text = df['clean_text']
        
        # Extract structural features
        X_structured = df[self.feature_cols].copy() if all(col in df.columns for col in self.feature_cols) else pd.DataFrame(index=df.index)
        
        # Extract stress-specific features
        stress_features = self.extract_stress_features(df['text'])
        
        # Add these new features to structural features
        for col in stress_features.columns:
            if col not in X_structured.columns:
                X_structured[col] = stress_features[col]
        
        # Handle missing columns
        for col in self.feature_cols:
            if col not in X_structured.columns:
                X_structured[col] = 0
        
        # Handle missing values with median
        for col in X_structured.columns:
            if X_structured[col].isnull().any():
                median_val = X_structured[col].median()
                X_structured[col] = X_structured[col].fillna(median_val)
        
        if training:
            # Improved TF-IDF with optimal parameters
            self.tfidf = TfidfVectorizer(
                max_features=2000,  # Increased from 1500
                min_df=2,           # Lower threshold to capture more rare terms
                max_df=0.9,         # Allow terms to appear in more documents
                ngram_range=(1, 3)  # Include trigrams to capture longer phrases
            )
            text_features = self.tfidf.fit_transform(X_text)
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            struct_features = self.scaler.fit_transform(X_structured)
        else:
            # Use pre-fitted vectorizer and scaler
            if self.tfidf is None or self.scaler is None:
                raise ValueError("Model components not loaded. Call train_model() or load_model() first")
            
            text_features = self.tfidf.transform(X_text)
            struct_features = self.scaler.transform(X_structured)
        
        # Combine features
        X_combined = np.hstack((text_features.toarray(), struct_features))
        
        return X_combined
    
    def train_model(self, df, model_type='logistic', grid_search=False):
        """Enhanced model training with class imbalance handling and optimized parameters."""
        logger.info(f"Training {model_type} model with improved parameters")
        
        # Get features and target
        X_combined = self.prepare_features(df, training=True)
        y = df['label']
        
        # Check for class imbalance
        class_counts = np.bincount(y)
        if class_counts[0] > 2 * class_counts[1] or class_counts[1] > 2 * class_counts[0]:
            logger.warning(f"Significant class imbalance detected: {class_counts}")
        
        # Use SMOTE for oversampling minority class if there's imbalance
        # (Note: You would need to add the imblearn library for this)
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_combined_resampled, y_resampled = smote.fit_resample(X_combined, y)
            logger.info(f"Applied SMOTE: {np.bincount(y)} → {np.bincount(y_resampled)}")
            X_combined, y = X_combined_resampled, y_resampled
        except ImportError:
            logger.warning("imblearn not available; proceeding without SMOTE")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Select model type with improved parameters
        if model_type == 'logistic':
            base_model = LogisticRegression(
                class_weight='balanced',
                C=1.0,  # Default regularization strength
                solver='liblinear',  # Works well with small/medium datasets
                max_iter=2000,  # Increased iterations for convergence
                random_state=42
            )
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],  # Wider range
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        elif model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=200,  # More trees
                class_weight='balanced',
                max_features='sqrt',  # Standard RF practice
                min_samples_leaf=2,  # Helps avoid overfitting
                n_jobs=-1,
                random_state=42
            )
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'svm':
            base_model = SVC(
                probability=True,
                class_weight='balanced',
                C=1.0,
                kernel='rbf',
                gamma='scale',
                random_state=42
            )
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model with enhanced cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if grid_search:
            logger.info("Performing grid search with optimized parameters")
            
            # Multi-metric scoring
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
            
            grid = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',  # Focus on F1 score as primary metric
                n_jobs=-1,
                return_train_score=True,
                verbose=1
            )
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            logger.info(f"Best parameters: {grid.best_params_}")
            logger.info(f"Best cross-validation score: {grid.best_score_:.3f}")
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)
        
        # Comprehensive model evaluation
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        logger.info(f"Test set metrics - ROC AUC: {roc:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model and components
        self.save_model()
        
        return report, cm, roc, y_test, y_proba
    
    def predict(self, text, structural_features=None):
        """Improved prediction with confidence calibration."""
        if self.model is None or self.tfidf is None or self.scaler is None:
            raise ValueError("Model components not loaded. Call train_model() or load_model() first")
        
        # Clean text
        clean = self.clean_text(text)
        text_vec = self.tfidf.transform([clean])
        
        # Extract stress-specific features directly from the text
        stress_features = self.extract_stress_features([text])
        
        # Handle structural features
        if structural_features is None:
            # Initialize with zeros
            struct_data = {col: 0 for col in self.feature_cols if col not in stress_features.columns}
            # Add stress features
            for col in stress_features.columns:
                struct_data[col] = stress_features[col].iloc[0]
            struct_vec = np.array([list(struct_data.values())])
        else:
            # Merge provided features with stress features
            if len(structural_features) != len(self.feature_cols) - len(stress_features.columns):
                # Handle the case where the length doesn't match
                provided_features = structural_features[:len(self.feature_cols) - len(stress_features.columns)]
                # Pad with zeros if needed
                if len(provided_features) < len(self.feature_cols) - len(stress_features.columns):
                    provided_features = list(provided_features) + [0] * (len(self.feature_cols) - len(stress_features.columns) - len(provided_features))
            else:
                provided_features = structural_features
                
            # Combine with stress features
            all_features = list(provided_features)
            for col in stress_features.columns:
                all_features.append(stress_features[col].iloc[0])
                
            struct_vec = np.array([all_features])
        
        # Scale features
        struct_vec = self.scaler.transform(struct_vec)
        
        # Combine features
        full_vec = np.hstack((text_vec.toarray(), struct_vec))
        
        # Get raw prediction and probability
        pred = self.model.predict(full_vec)[0]
        prob = self.model.predict_proba(full_vec)[0][1]
        
        # Apply stress-specific heuristic boost
        # If text contains strong suicide/stress indicators but probability is borderline,
        # we boost the confidence
        stress_indicators = stress_features['stress_word_count'].iloc[0]
        contains_suicide_terms = any(term in text.lower() for term in ['suicide', 'suicidal', 'kill myself', 'end my life'])
        
        # Boost probability if strong indicators are present but model is uncertain
        if (stress_indicators >= 3 or contains_suicide_terms) and 0.4 <= prob <= 0.6:
            logger.info(f"Applying boost to borderline prediction. Original prob: {prob:.3f}")
            # Adjust probability toward stress class
            boost_factor = 0.15 if contains_suicide_terms else 0.1
            prob = min(prob + boost_factor, 0.95)  # Cap at 0.95
            # Recalculate prediction based on boosted probability
            pred = 1 if prob > 0.5 else 0
            logger.info(f"Boosted probability: {prob:.3f}, New prediction: {pred}")
        
        return pred, prob
    
    def save_model(self):
        """Save model and components to disk."""
        logger.info(f"Saving model to {self.model_dir}")
        
        try:
            joblib.dump(self.model, os.path.join(self.model_dir, 'model.joblib'))
            joblib.dump(self.tfidf, os.path.join(self.model_dir, 'vectorizer.joblib'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self):
        """Load model and components from disk."""
        logger.info(f"Loading model from {self.model_dir}")
        
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'model.joblib'))
            self.tfidf = joblib.load(os.path.join(self.model_dir, 'vectorizer.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def compute_shap(self, df, sample_size=None):
        """Compute SHAP values for interpretation."""
        logger.info("Computing SHAP values")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call train_model() or load_model() first")
        
        # Sample data if needed
        if sample_size is not None and len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
        
        # Prepare features
        X_combined = self.prepare_features(df_sample, training=False)
        
        # Compute SHAP values
        try:
            explainer = shap.Explainer(self.model, X_combined)
            shap_values = explainer(X_combined)
            return explainer, shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise

    def plot_evaluation(self, y_test, y_proba):
        """Plot evaluation metrics including ROC and PR curves."""
        logger.info("Plotting evaluation metrics")
        
        try:
            # Create figure with multiple subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend()
            
            # Plot Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            
            ax2.plot(recall, precision)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.model_dir, 'evaluation_plots.png'))
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting evaluation metrics: {e}")
            return None

    def feature_importance(self):
        """Get feature importance from the model if available."""
        if self.model is None:
            raise ValueError("Model not loaded. Call train_model() or load_model() first")
        
        # Get feature names
        feature_names = self.tfidf.get_feature_names_out().tolist() + self.feature_cols
        
        # Extract importance based on model type
        if hasattr(self.model, 'coef_'):
            # Linear models
            importance = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
        else:
            logger.warning("Model doesn't provide feature importance")
            return None
        
        # Create DataFrame
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by absolute importance
        imp_df['abs_importance'] = abs(imp_df['importance'])
        imp_df = imp_df.sort_values('abs_importance', ascending=False)
        
        return imp_df


# Helper function to demonstrate usage
def run_model(file_path, model_type='logistic', grid_search=False):
    """Run the stress model pipeline."""
    model = StressModel()
    df = model.load_data(file_path)
    
    # Basic data analysis
    print(f"Dataset size: {len(df)} rows")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Train model
    report, cm, roc, y_test, y_proba = model.train_model(df, model_type, grid_search)
    
    # Plot evaluation metrics
    model.plot_evaluation(y_test, y_proba)
    
    # Feature importance
    importance = model.feature_importance()
    if importance is not None:
        print("\nTop 10 important features:")
        print(importance.head(10))
    
    # Sample prediction
    sample_text = df['text'].iloc[0]
    pred, prob = model.predict(sample_text)
    print(f"\nSample prediction for: {sample_text[:100]}...")
    print(f"Prediction: {'Stress' if pred == 1 else 'No Stress'}, Probability: {prob:.3f}")
    
    return model

if __name__ == '__main__':
    run_model('stress.csv', model_type='logistic', grid_search=True)
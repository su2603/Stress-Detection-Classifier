# Stress Detection System 

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [StressModel Class](#stressmodel-class)
   - [Features](#features)
   - [Core Methods](#core-methods)
   - [Model Training](#model-training)
   - [Prediction Pipeline](#prediction-pipeline)
4. [Streamlit Application](#streamlit-application)
   - [User Interface](#user-interface)
   - [Key Features](#key-features)
   - [Application Flow](#application-flow)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
   - [Training a New Model](#training-a-new-model)
   - [Loading a Pre-trained Model](#loading-a-pre-trained-model)
   - [Making Predictions](#making-predictions)
   - [Batch Processing](#batch-processing)
7. [Performance Metrics](#performance-metrics)
8. [Technical Details](#technical-details)
   - [Text Processing](#text-processing)
   - [Feature Extraction](#feature-extraction)
   - [Model Selection](#model-selection)
9. [Troubleshooting](#troubleshooting)
10. [Extension & Customization](#extension--customization)

## Overview

The Stress Detection System is a machine learning application designed to identify signals of stress, anxiety, and related mental health concerns in text data. Built using Python, scikit-learn, and Streamlit, the system provides both an algorithmic backend (StressModel) and a user-friendly web interface for model training, evaluation, and prediction.

This system can be used to:
- Analyze social media posts, messages, or journal entries for signs of stress
- Train custom stress detection models on domain-specific data
- Monitor stress trends in communication across time
- Identify individuals who may benefit from mental health resources

The application combines natural language processing techniques with machine learning to provide actionable insights from text data, with a particular focus on detecting linguistic patterns associated with stress and anxiety.

## System Architecture

The system consists of two main components:

1. **StressModel** (`stressModel.py`): A Python class that handles all machine learning operations including:
   - Data preprocessing and cleaning
   - Feature extraction (text and structured features)
   - Model training and evaluation
   - Prediction and interpretation

2. **Streamlit Application** (`stressPredApp.py`): A web interface that makes the model accessible to users without programming knowledge, offering:
   - Model training and loading capabilities
   - Interactive prediction tools
   - Performance visualization and evaluation
   - Batch processing options

The system follows a modular design where the model logic is separated from the presentation layer, allowing each component to be developed, maintained, and extended independently.

## StressModel Class

### Features

The `StressModel` class provides comprehensive functionality for stress detection:

- **Robust Text Processing**: Special handling for emotional indicators and stress-related linguistic patterns
- **Multi-modal Feature Extraction**: Combines text features with structured metadata
- **Model Flexibility**: Supports logistic regression, random forest, and SVM algorithms
- **Class Imbalance Handling**: Implements techniques to address uneven class distributions
- **Comprehensive Evaluation**: Provides detailed metrics and interpretability tools
- **Persistence**: Save and load trained models for future use
- **Explainability**: SHAP value computation for model interpretation

### Core Methods

| Method | Description |
|--------|-------------|
| `load_data` | Loads and preprocesses data from CSV files |
| `clean_text` | Applies specialized text cleaning while preserving stress indicators |
| `extract_stress_features` | Extracts stress-specific linguistic features |
| `prepare_features` | Combines text and structured features for model input |
| `train_model` | Trains machine learning models with optional grid search |
| `predict` | Makes predictions on new text with confidence calibration |
| `save_model` | Persists trained model and components to disk |
| `load_model` | Loads a previously trained model |
| `compute_shap` | Calculates SHAP values for model interpretation |
| `plot_evaluation` | Generates visualization of model performance |
| `feature_importance` | Extracts and ranks feature importance |

### Model Training

The model training pipeline includes:

1. Data loading and validation
2. Text cleaning and preprocessing
3. Feature extraction and normalization
4. Optional class imbalance handling using SMOTE
5. Model selection and training
6. Optional hyperparameter tuning via grid search
7. Comprehensive evaluation
8. Model persistence

The system supports three types of models:
- **Logistic Regression**: Effective for interpretability and baseline performance
- **Random Forest**: Robust to overfitting with good performance on diverse features
- **Support Vector Machine**: Good for complex decision boundaries with proper tuning

### Prediction Pipeline

The prediction process follows these steps:

1. Clean and preprocess input text
2. Extract stress-specific linguistic features
3. Apply the same feature extraction pipeline used during training
4. Generate predictions and confidence scores
5. Apply confidence calibration for borderline cases
6. Return the predicted class and probability

## Streamlit Application

### User Interface

The Streamlit application provides an intuitive interface organized into four tabs:

1. **Home**: Overview and system status
2. **Model Training**: Options for training new models or loading existing ones
3. **Evaluation**: Detailed performance metrics and visualizations
4. **Prediction**: Tools for analyzing single texts or batch processing

The interface features:
- Clean, modern design with custom CSS styling
- Interactive elements (sliders, input fields, file uploaders)
- Visual feedback on predictions and model performance
- Downloadable results for batch processing

### Key Features

- **Interactive Training**: Upload custom datasets and train models with different algorithms
- **Model Persistence**: Save and load trained models for continued use
- **Performance Visualization**: View confusion matrices, ROC curves, and precision-recall curves
- **Feature Importance**: Identify which features contribute most to predictions
- **Individual Text Analysis**: Enter text directly for immediate stress assessment
- **Batch Processing**: Upload multiple texts for efficient bulk analysis
- **Optional Feature Input**: Provide additional structured features to improve prediction accuracy

### Application Flow

The application follows this general flow:

1. User loads or trains a model
2. System provides performance metrics and visualizations
3. User can input text for analysis or upload a batch file
4. System processes the input and displays prediction results
5. User can iterate, trying different texts or models

## Installation & Setup

To install and run the Stress Detection System:

1. **Clone the repository or download the source files**
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk shap joblib streamlit
   ```
3. **Download required NLTK resources**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   nltk.download('omw-1.4')
   ```
4. **Run the Streamlit application**:
   ```bash
   streamlit run stressPredApp.py
   ```

## Usage Guide

### Training a New Model

To train a new model:

1. Navigate to the **Model Training** tab
2. Upload a CSV file containing your training data
   - Must include `text` and `label` columns
   - Optional structured feature columns
3. Select a model type (logistic, rf, svm)
4. Choose whether to perform grid search for hyperparameter tuning
5. Click "Train Model" and wait for the process to complete
6. Review model performance in the Evaluation tab

**Expected Data Format**:
- `text`: Text content to analyze
- `label`: Binary stress indicator (1 = stress, 0 = no stress)
- Optional features:
  - `sentiment`: Sentiment score
  - `syntax_fk_grade`: Flesch-Kincaid readability score
  - `syntax_ari`: Automated Readability Index
  - `social_karma`: User karma/reputation
  - `social_num_comments`: Number of comments/responses
  - `social_upvote_ratio`: Ratio of upvotes to total votes

### Loading a Pre-trained Model

To load a previously trained model:

1. Navigate to the **Model Training** tab
2. Enter the directory where your model is saved in the "Model directory" field
   - Default is "models"
3. Click "Load Model"
4. If successful, the system will confirm the model is loaded

### Making Predictions

To analyze text for stress signals:

1. Navigate to the **Prediction** tab
2. Enter text in the text area
3. Optionally, provide additional features using the "Add optional features" checkbox
4. Click "Analyze Text"
5. Review the results, including:
   - Prediction (Stress or No Stress)
   - Confidence level
   - Probability visualization

### Batch Processing

To process multiple texts at once:

1. Navigate to the **Prediction** tab
2. Prepare a CSV file with a `text` column
   - Optionally include structured feature columns
3. Upload the CSV file in the "Batch Predictions" section
4. Click "Run Batch Analysis"
5. Review the results table
6. Download the results using the "Download Results" button

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Visualization of true/false positives and negatives
- **Precision-Recall Curve**: Alternative to ROC for imbalanced data
- **Feature Importance**: Contribution of each feature to the model

## Technical Details

### Text Processing

The system employs specialized text processing to preserve stress indicators:

1. Lowercasing and basic cleaning
2. Special handling of emotional punctuation patterns (`!!!`, `???`, `...`)
3. Preserving negation words and stress indicators during stopword removal
4. Lemmatization to normalize word forms
5. Retention of emotional content that might indicate stress

### Feature Extraction

Features are extracted from two main sources:

1. **Text Features**:
   - TF-IDF vectorization with n-grams
   - Stress word frequency counts
   - Negative emotion ratios
   - Sentiment analysis

2. **Structured Features** (when available):
   - Readability metrics
   - Social engagement indicators
   - User reputation/karma
   - Comment counts and ratios

### Model Selection

The system offers three model types, each with strengths for different scenarios:

- **Logistic Regression**:
  - Good baseline performance
  - Highly interpretable coefficients
  - Works well with text features
  - Fast training and prediction

- **Random Forest**:
  - Robust to overfitting
  - Handles non-linear relationships
  - Built-in feature importance
  - Good with mixed feature types

- **Support Vector Machine**:
  - Effective in high-dimensional spaces
  - Memory efficient
  - Versatile through different kernels
  - Often works well for text classification

## Troubleshooting

**Common Issues and Solutions**:

1. **"No model loaded" warning**
   - Ensure you've trained a model or loaded a pre-trained model
   - Check that model files exist in the specified directory

2. **Error during training**
   - Verify your CSV has the required columns
   - Check for missing values in critical columns
   - Ensure labels are properly formatted (0/1)

3. **Poor model performance**
   - Consider providing more training data
   - Check for class imbalance in training data
   - Try different model types
   - Enable grid search for hyperparameter tuning

4. **Slow prediction or training**
   - Reduce the number of features
   - Use a smaller training dataset
   - Disable grid search for faster training
   - Use logistic regression for faster performance

5. **NLTK resource errors**
   - Manually download required resources
   - Check internet connection for download
   - Verify NLTK installation

## Extension & Customization

The system can be extended in several ways:

1. **Add new features**:
   - Modify `extract_stress_features` to include additional linguistic markers
   - Add new structured features to `feature_cols`

2. **Integrate new models**:
   - Add new model types to the `train_model` method
   - Include appropriate hyperparameter grids

3. **Enhance the UI**:
   - Modify the Streamlit app to add new visualizations
   - Add new tabs for additional functionality

4. **Implement advanced techniques**:
   - Add deep learning models for text classification
   - Implement more sophisticated NLP techniques
   - Add time series analysis for temporal patterns

5. **Integration with other systems**:
   - Develop APIs for service integration
   - Connect to data sources like social media platforms
   - Export results to notification systems or dashboards

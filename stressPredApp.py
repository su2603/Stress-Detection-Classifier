import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

# Import StressModel class
# If in same directory, otherwise adjust path
from stressModel import StressModel

# Page configuration
st.set_page_config(
    page_title="Stress Classifier",
    page_icon="😓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    body {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    .main-header {
        font-size: 2.7em;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
    }

    .subheader {
        font-size: 1.5em;
        color: #555;
        margin-bottom: 1.5em;
        text-align: center;
    }

    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 30px;
    }

    .prediction-stress {
        color: #d62728;
        font-weight: 700;
        font-size: 1.3em;
    }

    .prediction-no-stress {
        color: #2ca02c;
        font-weight: 700;
        font-size: 1.3em;
    }

    .confidence-high {
        color: #2ca02c;
        font-weight: 600;
    }

    .confidence-medium {
        color: #ff7f0e;
        font-weight: 600;
    }

    .confidence-low {
        color: #d62728;
        font-weight: 600;
    }

    div.stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #125b8c;
    }

    .result-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2ca02c;
        padding: 1em;
        border-radius: 8px;
        margin-top: 1em;
    }

    .stTextInput > div > input,
    .stTextArea > div > textarea {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'model' not in st.session_state:
        st.session_state.model = StressModel()
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'reports' not in st.session_state:
        st.session_state.reports = {}
    if 'confusion_matrix' not in st.session_state:
        st.session_state.confusion_matrix = None
    if 'roc_score' not in st.session_state:
        st.session_state.roc_score = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'y_proba' not in st.session_state:
        st.session_state.y_proba = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>Stress Classification App</h1>", unsafe_allow_html=True)
    st.markdown("This application helps identify stress signals in text using machine learning.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home", 
        "🧠 Model Training", 
        "📊 Evaluation", 
        "🔮 Prediction"
    ])
    
    # HOME TAB
    with tab1:
        st.markdown("<h2 class='subheader'>Welcome to the Stress Classifier!</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### About This Application
            
            This app uses machine learning to identify stress signals in text. It can be used to:
            
            - Analyze social media posts, journal entries, or messages for signs of stress
            - Train custom stress detection models on your own datasets
            - Understand what factors contribute most to stress identification
            
            ### Getting Started
            
            1. Go to the **Model Training** tab to train a new model or load a pre-trained one
            2. Check the **Evaluation** tab to see performance metrics
            3. Use the **Prediction** tab to analyze new text for stress indicators
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Model Status")
            
            if st.session_state.model_loaded:
                st.success("✅ Model is loaded and ready")
                if st.session_state.training_complete:
                    st.info("Model was trained this session")
                else:
                    st.info("Using pre-trained model")
            else:
                st.warning("⚠️ No model loaded")
                st.markdown("Go to the **Model Training** tab to load or train a model.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Features Used")
            
            st.markdown("""
            - Text content (TF-IDF features)
            - Sentiment scores
            - Readability metrics
            - Social engagement metrics
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # MODEL TRAINING TAB
    with tab2:
        st.markdown("<h2 class='subheader'>Model Training and Loading</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Train New Model")
            
            uploaded_file = st.file_uploader("Upload CSV training data", type=['csv'])
            
            model_type = st.selectbox(
                "Select model type",
                ["logistic", "rf", "svm"],
                help="Logistic Regression, Random Forest, or Support Vector Machine"
            )
            
            grid_search = st.checkbox(
                "Perform grid search for hyperparameter tuning", 
                value=False,
                help="This improves model performance but takes longer to train"
            )
            
            if uploaded_file is not None:
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        # Save uploaded file temporarily
                        temp_file_path = f"temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            # Load and train model
                            df = st.session_state.model.load_data(temp_file_path)
                            
                            # Show class distribution
                            st.write(f"Dataset size: {len(df)} rows")
                            class_dist = df['label'].value_counts(normalize=True) * 100
                            st.write(f"Class distribution: {class_dist[1]:.1f}% Stress, {class_dist[0]:.1f}% No Stress")
                            
                            # Train model
                            report, cm, roc, y_test, y_proba = st.session_state.model.train_model(
                                df, model_type, grid_search
                            )
                            
                            # Store results in session state
                            st.session_state.reports = report
                            st.session_state.confusion_matrix = cm
                            st.session_state.roc_score = roc
                            st.session_state.y_test = y_test
                            st.session_state.y_proba = y_proba
                            st.session_state.model_loaded = True
                            st.session_state.training_complete = True
                            
                            # Get feature importance
                            st.session_state.feature_importance = st.session_state.model.feature_importance()
                            
                            st.success("Model trained successfully!")
                            
                            # Clean up temp file
                            os.remove(temp_file_path)
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
            else:
                st.info("Please upload a CSV file to train the model")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Load Pre-trained Model")
            
            model_dir = st.text_input("Model directory", value="models")
            
            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    try:
                        # Initialize model with specified directory
                        st.session_state.model = StressModel(model_dir=model_dir)
                        
                        # Load model
                        st.session_state.model.load_model()
                        st.session_state.model_loaded = True
                        st.session_state.training_complete = False
                        
                        st.success("Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Expected Data Format")
            
            st.markdown("""
            Your CSV file should contain at least these columns:
            - `text`: The text content to analyze
            - `label`: Binary stress indicator (1 = stress, 0 = no stress)
            
            Optional feature columns:
            - `sentiment`: Sentiment score (numeric)
            - `syntax_fk_grade`: Flesch-Kincaid readability score
            - `syntax_ari`: Automated Readability Index
            - `social_karma`: User karma/reputation
            - `social_num_comments`: Number of comments/responses
            - `social_upvote_ratio`: Ratio of upvotes to total votes
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # EVALUATION TAB
    with tab3:
        st.markdown("<h2 class='subheader'>Model Evaluation</h2>", unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("Please load or train a model first!")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Performance Metrics")
                
                if st.session_state.reports:
                    # Format classification report
                    report = st.session_state.reports
                    
                    metrics_df = pd.DataFrame({
                        'Precision': [report['0']['precision'], report['1']['precision'], report['macro avg']['precision']],
                        'Recall': [report['0']['recall'], report['1']['recall'], report['macro avg']['recall']],
                        'F1-Score': [report['0']['f1-score'], report['1']['f1-score'], report['macro avg']['f1-score']],
                        'Support': [report['0']['support'], report['1']['support'], report['macro avg']['support']]
                    }, index=['No Stress (0)', 'Stress (1)', 'Macro Avg'])
                    
                    st.dataframe(metrics_df.style.format({
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-Score': '{:.3f}',
                        'Support': '{:.0f}'
                    }))
                    
                    # Overall accuracy
                    st.metric("Overall Accuracy", f"{report['accuracy']:.3f}")
                    
                    # ROC-AUC
                    if st.session_state.roc_score:
                        st.metric("ROC AUC Score", f"{st.session_state.roc_score:.3f}")
                else:
                    st.info("No evaluation metrics available. Train a model first.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Confusion Matrix
                if st.session_state.confusion_matrix is not None:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Confusion Matrix")
                    
                    cm = st.session_state.confusion_matrix
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['No Stress', 'Stress'],
                        yticklabels=['No Stress', 'Stress']
                    )
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # ROC and PR curves
                if st.session_state.y_test is not None and st.session_state.y_proba is not None:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### ROC & PR Curves")
                    
                    # Plot evaluation metrics
                    fig = st.session_state.model.plot_evaluation(
                        st.session_state.y_test, 
                        st.session_state.y_proba
                    )
                    
                    if fig:
                        st.pyplot(fig)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Feature importance
                if st.session_state.feature_importance is not None:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Feature Importance")
                    
                    # Get top features
                    importance = st.session_state.feature_importance
                    top_n = min(15, len(importance))
                    top_features = importance.head(top_n)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = ['#3366cc' if x >= 0 else '#cc3333' for x in top_features['importance']]
                    
                    plot = sns.barplot(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        ax=ax,
                        errorbar=None,  # Disable error bars
                        orient='h'      # Horizontal orientation
                    )
                    
                    # Set bar colors manually to avoid hue-related warnings
                    for i, bar in enumerate(plot.containers[0]):
                        importance_value = top_features['importance'].iloc[i]
                        bar.set_color('#3366cc' if importance_value >= 0 else '#cc3333')
                    
                    plt.title('Top Feature Importance')
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # PREDICTION TAB
    with tab4:
        st.markdown("<h2 class='subheader'>Make Predictions</h2>", unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("Please load or train a model first!")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Enter Text for Analysis")
                
                # Text input
                user_text = st.text_area(
                    "Enter text to analyze for stress signals",
                    height=200,
                    help="Enter a message, social media post, or any text to analyze"
                )
                
                # Feature inputs (optional)
                show_features = st.checkbox("Add optional features", value=False)
                
                feature_values = None
                if show_features:
                    st.markdown("### Optional Features")
                    st.markdown("These additional metrics can improve prediction accuracy.")
                    
                    col1a, col1b = st.columns(2)
                    with col1a:
                        sentiment = st.slider(
                            "Sentiment score (-1 to 1)", 
                            min_value=-1.0, 
                            max_value=1.0, 
                            value=0.0,
                            help="Sentiment analysis score: -1 (negative) to 1 (positive)"
                        )
                        
                        fk_grade = st.slider(
                            "Flesch-Kincaid Grade Level", 
                            min_value=0.0, 
                            max_value=20.0, 
                            value=8.0,
                            help="Readability score (higher means more complex text)"
                        )
                        
                        ari = st.slider(
                            "Automated Readability Index", 
                            min_value=0.0, 
                            max_value=20.0, 
                            value=8.0,
                            help="Alternative readability metric"
                        )
                        
                    with col1b:
                        karma = st.number_input(
                            "User Karma/Reputation", 
                            min_value=0, 
                            value=100,
                            help="Social media karma or reputation score"
                        )
                        
                        comments = st.number_input(
                            "Number of Comments", 
                            min_value=0, 
                            value=5,
                            help="Number of comments or responses"
                        )
                        
                        upvote_ratio = st.slider(
                            "Upvote Ratio", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.75,
                            help="Ratio of upvotes to total votes (0.5 = neutral)"
                        )
                    
                    feature_values = [sentiment, fk_grade, ari, karma, comments, upvote_ratio]
                
                # Prediction button
                if st.button("Analyze Text"):
                    if not user_text:
                        st.error("Please enter some text to analyze.")
                    else:
                        with st.spinner("Analyzing text..."):
                            try:
                                # Make prediction
                                pred, prob = st.session_state.model.predict(user_text, feature_values)
                                
                                # Display result
                                st.markdown("### Analysis Result")
                                
                                result_col1, result_col2 = st.columns(2)
                                
                                with result_col1:
                                    if pred == 1:
                                        st.markdown("<p class='prediction-stress'>✋ Stress Detected</p>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("<p class='prediction-no-stress'>👍 No Stress Detected</p>", unsafe_allow_html=True)
                                
                                with result_col2:
                                    # Confidence level
                                    confidence = max(prob, 1-prob)
                                    if confidence > 0.8:
                                        st.markdown(f"<p class='confidence-high'>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
                                    elif confidence > 0.6:
                                        st.markdown(f"<p class='confidence-medium'>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<p class='confidence-low'>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
                                
                                # Probability gauge
                                fig, ax = plt.subplots(figsize=(8, 2))
                                ax.barh(y=0, width=1, color='lightgray')
                                ax.barh(y=0, width=prob, color='#ff7f0e' if pred == 1 else '#2ca02c')
                                ax.set_yticks([])
                                ax.set_xlim(0, 1)
                                ax.set_xlabel('Probability of Stress')
                                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Batch Predictions")
                
                st.markdown("Upload a CSV file with texts to analyze in bulk.")
                
                batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
                
                if batch_file is not None:
                    # Process batch file
                    if st.button("Run Batch Analysis"):
                        with st.spinner("Processing batch predictions..."):
                            try:
                                # Read CSV file
                                batch_df = pd.read_csv(batch_file)
                                
                                if 'text' not in batch_df.columns:
                                    st.error("CSV must contain a 'text' column")
                                else:
                                    # Check for structural features
                                    has_features = all(col in batch_df.columns for col in st.session_state.model.feature_cols)
                                    
                                    # Make predictions
                                    results = []
                                    for i, row in batch_df.iterrows():
                                        text = row['text']
                                        
                                        if has_features:
                                            features = row[st.session_state.model.feature_cols].values.tolist()
                                            pred, prob = st.session_state.model.predict(text, features)
                                        else:
                                            pred, prob = st.session_state.model.predict(text)
                                        
                                        results.append({
                                            'text': text[:100] + '...' if len(text) > 100 else text,
                                            'prediction': 'Stress' if pred == 1 else 'No Stress',
                                            'probability': prob
                                        })
                                    
                                    # Show results
                                    results_df = pd.DataFrame(results)
                                    st.dataframe(results_df.style.format({
                                        'probability': '{:.3f}'
                                    }))
                                    
                                    # Download results
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        "Download Results",
                                        csv,
                                        "stress_predictions.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                            
                            except Exception as e:
                                st.error(f"Error processing batch: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Interpretation Tips")
                
                st.markdown("""
                **Understanding the results:**
                
                - High confidence (>80%) indicates a strong signal
                - Medium confidence (60-80%) suggests potential stress indicators
                - Low confidence (<60%) means the model is uncertain
                
                **Common stress indicators in text:**
                
                - Negative sentiment words
                - Expressions of worry or anxiety
                - Mentions of time pressure or deadlines
                - Words related to health concerns
                - Language indicating feeling overwhelmed
                """)
                
                st.markdown("</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
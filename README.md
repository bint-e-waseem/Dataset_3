# Toxic Comments Classification - EDA & Machine Learning Analysis

## 📊 Project Overview

This project presents a comprehensive Exploratory Data Analysis (EDA) and Machine Learning pipeline for classifying toxic comments across multiple categories. The dataset contains 30,000 comments labelled across five toxicity categories: Clean, Toxic, Severe_Toxic, Threat, and Identity_Attack.

## 🔍 Key Insights

### 1. Dataset Composition
- **Total Samples**: 30,000 comments
- **Target Distribution**:
  - Clean: 61.93% (18,579 samples)
  - Toxic: 22.31% (6,693 samples)
  - Severe_Toxic: 6.00% (1,801 samples)
  - Identity_Attack: 4.89% (1,468 samples)
  - Threat: 4.86% (1,459 samples)

### 2. Text Analysis Findings
- **Average Comment Length**: 56 characters
- **Average Word Count**: 8.3 words
- **Most Common Topics**: Technology, Education, Gaming, Politics, Sports, Entertainment
- **Platform Distribution**: Community Chat, News Comments, Blog, Social Media, Forum

### 3. Correlation Insights
- Strong correlation between Toxic and Severe_Toxic labels
- Identity_Attack shows correlation with Threat labels
- Clean labels are negatively correlated with all toxic categories

### 4. Feature Engineering
- **TF-IDF Features**: 132 unique features extracted
- **Sentiment Analysis**: Polarity and subjectivity scores added
- **Text Features**: Word count, character count, special character counts

## 🛠️ Technical Implementation

### Data Preprocessing
- Text cleaning (lowercase, URL removal, special character removal)
- Tokenisation and lemmatisation using NLTK
- Stopword removal
- TF-IDF vectorisation with n-gram range (1,2)

### Feature Engineering
- Comment length metrics
- Punctuation counting (exclamation marks, question marks)
- Uppercase character count
- Sentiment polarity and subjectivity (via TextBlob)

### Model Performance
- **Model**: Logistic Regression (Baseline)
- **Features**: TF-IDF vectors + Numeric features
- **Performance**: High accuracy achieved on test set

## 📈 Visualizations

The analysis includes:
1. Toxicity label distribution (Bar chart & Pie chart)
2. Word cloud of most frequent terms
3. Correlation matrix of binary labels
4. Platform and topic distribution
5. Text length distribution by toxicity category

## 🚀 Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```python
# Load and preprocess data
df = pd.read_csv('toxic_comments_dataset.csv')

# Run EDA notebook
jupyter notebook EDA_diabetes.ipynb
```

## 📚 Dependencies
- pandas, numpy for data manipulation
- scikit-learn for ML modelling
- nltk for text preprocessing
- matplotlib, seaborn for visualisation
- textblob for sentiment analysis

## 🔮 Future Work
- Implement deep learning models (LSTM, BERT)
- Explore ensemble methods
- Deploy model as an API service
- Real-time comment moderation system

## 📝 License
This project is for educational and research purposes.

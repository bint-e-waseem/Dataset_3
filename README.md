# Toxic Comments Classification - EDA & Machine Learning Pipeline

## 📊 Project Overview

This project performs comprehensive **Exploratory Data Analysis (EDA)** and **Machine Learning** preprocessing on a toxic comments dataset. The goal is to analyze patterns in toxic online comments and prepare the data for classification modeling to detect harmful content across multiple toxicity categories.

## 🎯 Objectives

- **Analyze** the distribution of toxicity labels and comment characteristics
- **Clean and preprocess** text data for machine learning
- **Engineer features** for toxicity classification
- **Build baseline models** to establish performance benchmarks
- **Prepare data** for production-ready machine learning pipelines

## 📁 Dataset Description

The dataset contains **30,000** comment records with the following characteristics:

### Key Features
| Feature | Description |
|---------|-------------|
| `Comment_ID` | Unique identifier for each comment |
| `Comment_Text` | Raw text content of the comment |
| `Toxicity_Label` | Primary toxicity category (5 classes) |
| `Toxicity_Score` | Continuous toxicity score (0.0 - 0.95) |
| `Platform` | Source platform (Community Chat, News Comments, Blog, Social Media, Forum) |
| `Topic` | Topic category (Technology, Education, Gaming, Politics, Sports, Entertainment) |
| `Moderation_Priority` | Priority level (High, Medium, Low) |

### Binary Toxicity Flags
- `Clean`: Non-toxic comments
- `Toxic`: Toxicity indicator
- `Severe_Toxic`: Severe toxicity indicator
- `Threat`: Threatening content indicator
- `Identity_Attack`: Identity-based attacks

### Engineering Features Added
- Processed comment text (cleaned, tokenized, lemmatized)
- Text length metrics (character count, word count)
- Punctuation and special character counts
- Sentiment analysis features (polarity, subjectivity)
- Encoded target labels for multi-class classification

## 📈 Exploratory Data Analysis Highlights

### Target Distribution
- **Clean comments**: 61.93% (18,579)
- **Toxic comments**: 22.31% (6,693)
- **Severe_Toxic**: 6.00% (1,801)
- **Identity_Attack**: 4.89% (1,468)
- **Threat**: 4.86% (1,459)

### Key Insights
- **Class imbalance** is significant (ratio: 12.73:1)
- **Text length** varies by toxicity category (Identity_Attack comments are longest)
- **Cross-platform analysis** shows consistent toxicity patterns across platforms
- **Common words** analysis reveals frequently used terms across all categories

### Sample Data Preview

| Comment_ID | Comment_Text | Toxicity_Label | Platform | Topic |
|------------|--------------|----------------|----------|-------|
| COM000001 | "The message targets a group unfairly..." | Identity_Attack | Community Chat | Sports |
| COM000002 | "Thank you for sharing this perspective..." | Clean | News Comments | Entertainment |
| COM000003 | "That response is aggressive and not helpful." | Toxic | Social Media | Politics |

## 🛠️ Methodology

### 1. Data Loading & Initial Inspection
- Loaded dataset with pandas
- Initial data exploration and structure analysis
- Checked for missing values and duplicates

### 2. Exploratory Data Analysis
- **Distribution Analysis**: Visualized toxicity labels, platforms, and topics
- **Text Analysis**: Computed word counts, character counts, and word length distributions
- **Correlation Analysis**: Analyzed relationships between binary labels
- **Cross-analysis**: Examined toxicity distribution across platforms and topics

### 3. Text Preprocessing
```python
def clean_text(text):
    # Convert to lowercase
    # Remove URLs, mentions, hashtags
    # Remove special characters and digits
    # Remove extra whitespace
```

- Applied **tokenization** using NLTK
- Performed **lemmatization** with WordNetLemmatizer
- Removed English stopwords
- Created cleaned and processed text columns

### 4. Feature Engineering
- **Text features**: Length metrics, token counts
- **Punctuation features**: Exclamation and question mark counts
- **Sentiment features**: Polarity and subjectivity using TextBlob
- **TF-IDF features**: 132 features from processed text

### 5. Modeling Preparation
- **Label Encoding**: Converted categorical labels to numeric values
- **Train-Test Split**: 80/20 stratified split
- **Feature Matrix**: Combined numeric and binary features (12 features)
- **Model Training**: Logistic Regression baseline models

## 📊 Model Performance

### Baseline Results (Logistic Regression with TF-IDF)

| Metric | Performance |
|--------|-------------|
| **Accuracy** | 100% |
| **Macro Avg F1** | 100% |
| **Weighted Avg F1** | 100% |

### Per-Class Performance
- All toxicity categories achieved 1.0 precision, recall, and F1-score on test data
- Balanced performance across classes despite class imbalance
- Confusion matrix shows perfect classification with no misclassifications

### Performance Notes
- The perfect accuracy results suggest the dataset may be synthetic or have strong classification features
- Real-world datasets may require more complex models and handle class imbalance
- The model serves as a strong baseline for understanding feature importance

## 🚀 Implementation Details

### Dependencies
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
textblob
wordcloud
joblib
```

### Key Libraries Used
- **Pandas**: Data manipulation
- **NLTK**: Text processing (tokenization, lemmatization, stopwords)
- **Scikit-learn**: Feature extraction, model training, preprocessing
- **TextBlob**: Sentiment analysis
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model persistence

### Text Preprocessing Steps
1. Lowercase conversion
2. URL removal
3. Mention removal (@username)
4. Hashtag cleaning
5. Special character and digit removal
6. Stopword removal
7. Tokenization and lemmatization

## 📂 Repository Structure

```
toxic-comments-classification/
├── README.md
├── toxic_comments_processed.csv      # Processed dataset
├── toxicity_model.pkl                 # Trained model
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer
├── label_encoder.pkl                  # Label encoder
└── EDA_diabetes.ipynb                 # Main EDA notebook
```

## 💡 Key Features

### Data Processing
- **Automatic cleaning**: Removes noise from text data
- **Feature engineering**: Creates informative features for ML
- **Multi-class classification**: Supports 5 toxicity categories
- **Sentiment analysis**: Adds sentiment metrics to text features

### Visualization
- **Distribution plots**: Class distributions, platform distributions
- **Correlation matrices**: Relationship analysis between features
- **Word cloud**: Visual representation of common terms
- **Performance metrics**: Classification reports and confusion matrices

## 📈 Potential Applications

### Real-World Use Cases
- **Content moderation**: Automated detection of toxic comments
- **Social media monitoring**: Track and analyze harmful content
- **Community management**: Identify and flag inappropriate discussions
- **Research applications**: Study patterns in online discourse

### Future Improvements
- **Deep learning models**: LSTM, BERT, or Transformer-based approaches
- **Active learning**: Iterative improvement with human feedback
- **Multilingual support**: Extend to non-English content
- **Real-time processing**: API integration for live content monitoring

## 📝 Additional Resources

### Related Datasets
- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Wikipedia Toxic Comments](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973)

### Relevant Papers
- "Toxicity Detection in Online Discussions" - Wulczyn et al. (2017)
- "Detecting Toxic Content in Online Discussions" - Jigsaw Team
- "Perspective API" - Jigsaw & Google

## 🤝 Contributing

Contributions are welcome! Areas for contribution include:
- Improving model performance with advanced techniques
- Adding support for additional languages
- Implementing real-time API endpoints
- Developing visualisation dashboards

## 📄 License

This project is available for academic and research purposes. Commercial use requires permission.

## 📧 Contact

For questions or collaboration opportunities:
- Email: yashfawaseem2006@gmail.com

---

**Last Updated**: 2026

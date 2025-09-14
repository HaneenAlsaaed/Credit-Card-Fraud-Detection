# Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

A comprehensive machine learning project focused on detecting fraudulent credit card transactions using various classification algorithms and advanced data analysis techniques.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Models Implemented](#-models-implemented)
- [Key Features](#-key-features)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)

## 🎯 Project Overview

Credit card fraud is a significant concern in the financial industry, causing billions of dollars in losses annually. This project implements and compares multiple machine learning algorithms to accurately detect fraudulent transactions while minimizing false positives that could inconvenience legitimate customers.

### Objectives
- Develop robust machine learning models for fraud detection
- Handle severely imbalanced datasets effectively
- Compare performance across multiple algorithms
- Optimize models for real-world deployment scenarios
- Provide comprehensive analysis and visualization of results

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle, which contains:

- **284,807 transactions** made by European cardholders in September 2013
- **492 fraudulent transactions** (0.172% of all transactions)
- **30 features** including:
  - `V1-V28`: Principal Component Analysis (PCA) transformed features (anonymized)
  - `Time`: Seconds elapsed between transactions
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Normal, 1 = Fraud)

### Dataset Characteristics
- **Highly imbalanced**: Only 0.172% of transactions are fraudulent
- **Privacy-protected**: Sensitive features are PCA transformed
- **Real-world data**: Actual credit card transactions from European cardholders

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb    # Main analysis notebook
├── Group-2(VISIONARIES)-Final Presentation.pptx.pptx  # Project presentation
├── SIC_AI_Capstone Project_Final Report.docx          # Detailed project report
├── SIC_AI_Capstone Project_Work Breakdown Structure.xlsx  # Project planning
└── README.md                           # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm kagglehub
```

### Dependencies List
```python
# Data manipulation and analysis
pandas
numpy

# Visualization
matplotlib
seaborn

# Machine learning
scikit-learn
imbalanced-learn

# Gradient boosting
xgboost
lightgbm

# Dataset download
kagglehub
```

## 💻 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HaneenAlsaaed/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```

4. **Run the analysis:**
   - Execute cells sequentially to reproduce the complete analysis
   - The notebook will automatically download the dataset using Kaggle API

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Analysis**: Comprehensive check for missing data
- **Duplicate Detection**: Identification and removal of duplicate transactions
- **Statistical Analysis**: Descriptive statistics and distribution analysis
- **Feature Scaling**: StandardScaler for numerical features

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution Analysis**: Visualization of fraud vs. normal transactions
- **Transaction Amount Analysis**: Distribution patterns by transaction type
- **Temporal Analysis**: Transaction patterns over time
- **Correlation Analysis**: Feature relationship exploration
- **Advanced Visualizations**: Heatmaps, distribution plots, and time series analysis

### 3. Data Preparation
- **Stratified Train-Test Split**: Maintaining class distribution
- **Feature Scaling**: Standardization of numerical features
- **Imbalanced Data Handling**: Multiple resampling techniques
- **Cross-Validation Setup**: Stratified K-Fold validation

### 4. Model Development
- **Baseline Models**: Implementation of multiple algorithms
- **Hyperparameter Tuning**: Optimization for best performance
- **Threshold Optimization**: Adjusting decision thresholds for optimal precision-recall balance
- **Cross-Validation**: Robust model evaluation

## 🤖 Models Implemented

### 1. Logistic Regression
- **Baseline Model**: Simple linear classifier
- **Balanced Version**: Using class weights
- **With Resampling**: SMOTE, Random Over/Under-sampling
- **Threshold Tuning**: Optimized for fraud detection

### 2. Random Forest
- **Ensemble Method**: Multiple decision trees
- **Feature Importance**: Identification of key fraud indicators
- **Balanced Classes**: Handling imbalanced data
- **Robust Performance**: Good generalization capabilities

### 3. XGBoost
- **Gradient Boosting**: Advanced ensemble technique
- **Scale Position Weight**: Handling class imbalance
- **High Performance**: State-of-the-art results
- **Feature Selection**: Built-in feature importance

### 4. LightGBM
- **Efficient Gradient Boosting**: Fast and memory-efficient
- **Advanced Optimization**: Hyperparameter tuning
- **Threshold Adjustment**: Precision-recall optimization
- **Scalable Solution**: Suitable for large datasets

### 5. Decision Tree
- **Interpretable Model**: Clear decision rules
- **Feature Analysis**: Easy-to-understand splits
- **Baseline Comparison**: Simple yet effective approach

## ✨ Key Features

### Advanced Techniques
- **Multiple Resampling Methods**: SMOTE, Random Over/Under-sampling
- **Stratified Cross-Validation**: Robust model evaluation
- **Threshold Optimization**: Balanced precision-recall trade-off
- **Feature Importance Analysis**: Understanding fraud indicators
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Handling Imbalanced Data
- **Class Weight Balancing**: Algorithmic approach to handle imbalance
- **SMOTE**: Synthetic minority oversampling technique
- **Random Sampling**: Over and under-sampling strategies
- **Evaluation Metrics**: Focus on precision and recall for minority class

### Visualization and Analysis
- **Distribution Plots**: Class and feature distributions
- **Correlation Heatmaps**: Feature relationship analysis
- **ROC Curves**: Model performance visualization
- **Precision-Recall Curves**: Threshold optimization
- **Confusion Matrices**: Detailed classification results

## 📈 Results

### Model Performance Summary

| Model | Technique | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|-----------|--------|----------|---------|
| Logistic Regression | SMOTE | High | Good | Balanced | Strong |
| Random Forest | Balanced | Excellent | Good | Strong | Excellent |
| XGBoost | Oversampling | Excellent | High | Strong | Excellent |
| LightGBM | Tuned | Excellent | High | Strong | Excellent |
| Decision Tree | Baseline | Good | Moderate | Moderate | Good |

### Key Insights
- **XGBoost and LightGBM** achieved the highest performance
- **Random Forest with balanced classes** showed excellent precision
- **SMOTE technique** effectively improved minority class detection
- **Threshold tuning** significantly enhanced fraud detection rates

## 📊 Visualizations

The project includes comprehensive visualizations:

### Data Analysis
- Class distribution bar charts
- Transaction amount distributions
- Time-based transaction patterns
- Feature correlation heatmaps

### Model Performance
- ROC curves for all models
- Precision-recall curves
- Confusion matrices
- Feature importance plots
- Model comparison charts

### Advanced Analytics
- Transaction patterns over time
- Fraud vs. normal transaction characteristics
- Feature relationship analysis
- Performance metric comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Team

**Group 2 - VISIONARIES**

This project was developed as part of the SIC AI Capstone Project, demonstrating advanced machine learning techniques for fraud detection in financial transactions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the Credit Card Fraud Detection dataset
- **SIC AI Program** for the learning opportunity and project framework
- **Open Source Community** for the excellent machine learning libraries used in this project

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please feel free to reach out through GitHub issues or create a discussion.

---

**Note**: This project is for educational and research purposes. When implementing fraud detection systems in production, additional considerations such as real-time processing, data privacy, regulatory compliance, and continuous model monitoring should be addressed.
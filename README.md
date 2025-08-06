# 💳 Loan Default Prediction System  

A system designed to predict loan default risk using advanced Machine Learning and Deep Learning models. The goal is to assist financial institutions in assessing creditworthiness and reducing the risk of non-performing loans.

## 📌 Objective  
This system processes loan application data, cleans and balances it, and applies classification models to determine the likelihood of loan repayment. It provides accuracy metrics and classification reports to support informed decision-making.

## ✅ Scope  
- Data preprocessing: handling missing values, encoding categorical variables, and scaling features  
- Class imbalance correction using SMOTE  
- Training and evaluation of three models:
  - Random Forest Classifier  
  - Perceptron  
  - Feedforward Neural Network  
- Performance reporting with accuracy scores and classification metrics  

## 🧠 Workflow Overview  
1. **Data Loading**  
   Load `loan_data.csv`, containing customer loan information.  
2. **Preprocessing & Balancing**  
   - Handle missing values with median imputation  
   - Encode categorical values using LabelEncoder  
   - Standardize features with StandardScaler  
   - Balance dataset with SMOTE  
3. **Model Training**  
   - Random Forest Classifier  
   - Perceptron  
   - Feedforward Neural Network (Keras)  
4. **Evaluation**  
   Output classification reports and accuracy for all models  

⚙ How It Works  
1. Place `loan_data.csv` in the project root.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt


## Structure   
├── Loan.py          # Main script for training & evaluation  
├── loan_data.csv    # Dataset file (user-provided)  
├── requirements.txt # Dependencies  
└── README.md        # Documentation  

Dataset : https://www.kaggle.com/datasets/urstrulyvikas/house-loan-data-analysis

# credit_card_fraud_detection


## Overview
This project aims to detect fraudulent credit card transactions using various machine learning models, including traditional algorithms and neural networks. The goal is to accurately identify potentially fraudulent transactions.

## Data Preprocessing
1. **Data Loading**: The dataset was loaded from CSV files ('fraudTrain.csv' and 'fraudTest.csv').
2. **Class Balancing**: Due to the imbalanced nature of fraud detection datasets, downsampling of the majority class (non-fraudulent transactions) was performed to match the minority class (fraudulent transactions).
3. **Feature Selection**: Unnecessary columns were removed, including 'Unnamed: 0', 'merchant', 'cc_num', 'first', 'zip', 'last', 'trans_num', 'unix_time', 'street', 'merch_lat', 'gender', 'merch_long', 'job', and 'trans_date_trans_time'.
4. **Missing Value Handling**: 
   - Numerical features: Filled with mean or median values.
   - Categorical features: Filled with mode values.
   - 'dob' column: Rows with missing values were dropped.
5. **Duplicate Removal**: Duplicate entries were removed from the dataset.
6. **Feature Engineering**: 
   - 'dob' was converted to 'age' by calculating the difference from the current year.
7. **Outlier Handling**: Outliers in numerical features were capped using the Interquartile Range (IQR) method.
8. **Encoding**: Categorical features were encoded using Label Encoding.
9. **Scaling**: Numerical features were standardized using StandardScaler.

## Model Development and Evaluation

### Traditional Machine Learning Models
Several traditional machine learning models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)

#### Initial Model Accuracies:
- Logistic Regression: 0.849938
- Decision Tree: 0.948339
- Random Forest: 0.958180
- Gradient Boosting: 0.953260
- KNN: 0.846248

### Hyperparameter Tuning
Hyperparameter tuning was performed using both GridSearchCV and RandomizedSearchCV to optimize the performance of each model.

### Neural Network Model
A deep neural network was implemented using TensorFlow/Keras with the following characteristics:
- Multiple dense layers with ReLU activation
- Batch Normalization and Dropout for regularization
- Binary cross-entropy loss and Adam optimizer
- Learning rate scheduling and early stopping

#### Neural Network Performance:
- Accuracy: 0.9237
- Precision: 0.9123
- Recall: 0.9309
- F1-Score: 0.9215
- AUC-ROC: 0.9768
- False Positive Rate: 0.0829
- False Negative Rate: 0.0691

### Hyperparameter Tuning for Neural Network
Keras Tuner was used to perform hyperparameter optimization for the neural network, exploring different architectures and learning rates.

## Conclusion
This project demonstrates the effectiveness of various machine learning approaches in detecting credit card fraud. The Random Forest model showed the highest accuracy among traditional models, while the neural network achieved strong performance across multiple metrics. The combination of careful preprocessing, feature engineering, and model optimization contributed to the success of the fraud detection system.

## Future Work
- Explore more advanced feature engineering techniques
- Implement ensemble methods combining multiple models
- Investigate the use of anomaly detection algorithms
- Conduct more extensive hyperparameter tuning
- Test the models on larger and more diverse datasets

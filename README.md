# Mileage-predictor-ML
A machine learning-based mileage prediction system built with Python. Implements data preprocessing, model training using Linear and Random Forest regression, and evaluation. The trained model is serialized with Pickle and deployed via a streamlit interface for real-time predictions.


 Features
- Data Preprocessing: Cleans and transforms raw vehicle data (e.g., handling missing values, encoding categorical variables, scaling numerical features).
- Model Training: Implements both Linear Regression and Random Forest Regression for predicting vehicle mileage.
- Model Evaluation: Assesses performance using MAE, RMSE, and R² Score.
- Model Serialization: Saves the trained model using Pickle for fast loading and deployment.
- Streamlit Interface: Provides a user-friendly web interface for real-time predictions based on user input.

Tech Stack:
- Programming Language: Python
- ML Libraries: scikit-learn, pandas, numpy
- Visualization: matplotlib, seaborn
- Model Persistence: pickle
- Web Interface: streamlit
- Environment Manager: conda

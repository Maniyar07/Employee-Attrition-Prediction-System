# Employee-Attrition-Prediction-System
Employee Attrition Prediction is a machine learning project designed to identify whether an employee is likely to leave an organization. This project includes a complete end-to-end 
ML workflow: data preprocessing, feature engineering, model training, evaluation, and 
deployment using a Flask web application.

The model predicts attrition based on multiple factors such as job role, experience, 
work environment satisfaction, salary hike, overtime, and other HR-related features.

### 🔥 Key Features
- End-to-end ML pipeline with clean and modular code
- Data preprocessing using label encoding & scaling
- Model training with algorithms like Random Forest / Logistic Regression
- Performance metrics: accuracy, precision, recall, F1-score
- Saving trained model using pickle
- Flask-based web interface for real-time predictions
- User-friendly form to input employee details
- Ready for deployment on PythonAnywhere / Render

### 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Flask  
- HTML/CSS  
- Pickle for model serialization

### 📂 Project Structure
- `app.py` – Flask application
- `model.py` – trained ML model
- `model.pkl` –An intermediate or backup serialized ML model used during development or testing.
- `trained_model.pkl` – The final, production-ready model loaded by app.py for real-time predictions.
- `encoders.pkl` – saved label encoders
- `columns.pkl` –A serialized list ensuring the correct feature order and set for new data before prediction.
- `templates/` – HTML files
- `static/` – CSS files


### 🎯 Purpose
This project is designed to help HR teams take data-driven decisions, reduce employee turnover, 
and understand key factors contributing to attrition. It is an excellent portfolio project for 
machine learning beginners and job seekers.


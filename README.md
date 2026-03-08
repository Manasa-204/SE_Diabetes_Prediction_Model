Diabetes Prediction using Machine Learning and AI 

Description:
This project predicts the likelihood of diabetes using machine learning based on medical attributes such as glucose level, BMI, age, and insulin.
The model is trained on a healthcare dataset and evaluates prediction accuracy using classification algorithms.

Tech Stack:
Python
Scikit-learn
Pandas
NumPy
Matplotlib
FlaskAPI
PHPAdmin

Features:
Data preprocessing
Model training
Accuracy evaluation
Diabetes risk prediction

How to Run:
1. Clone the repository
2. Install requirements
Three folders are present to run all venv is must

Steps :
1]Check python version - preferably above 3.8+
  python --version
2]Inside the main project folder
  python -m venv venv
3]To activate : venv\Scripts\activate
4]Install requirements 
  pip install -r requirements.txt 
5]Deactive : deactivate

After configuring everything if any more dependencies required , add accordingly
Steps to run the project:
1]Active venv
  .\venv\Scripts\activate
2]Go into the diabetes_backend folder
  run : uvicorn app:app --reload
  The obtained link can be hosted on browser to check
  also along with host link/docs gives the physical environment
2]Go into the diabetes_frontend-ai folder
  npm install
  npm run
  Follow the link to use the website on your browser

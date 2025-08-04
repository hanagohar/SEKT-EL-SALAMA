# 🚧 Traffic Accident Prediction System

A machine learning project that predicts the likelihood of a traffic accident using real-world road and traffic data. The system is designed to improve road safety, assist traffic management, and reduce emergency response time.

## 💡 Overview

We used a dataset from Kaggle:  
**Smart Mobility and Traffic Optimization Dataset**  
It includes data on traffic speed, vehicle count, road occupancy, weather, accidents, ride-sharing, and traffic lights.

## 🧠 Features Used
- Traffic speed
- Number of vehicles
- Road occupancy
- Weather condition
- Traffic light state
- Time-based features (hour, weekday, rush hour)

## 🧪 Data Science Workflow
1. **Data Collection & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Model Training**
5. **Evaluation**
6. **Deployment via GUI**

## 🔍 Models Used
- Random Forest  
- XGBoost (with GridSearchCV)  
- SVM  
- Combined with a Voting Classifier  

## 🎯 Performance
- Evaluation metrics: Accuracy, F1-score  
- Feature importance analyzed  
- Used SMOTE for class balancing  
- PowerTransformer for skewed features  

## 🖥️ GUI
A user-friendly interface built to:
- Take real-time inputs (vehicle count, speed, weather, etc.)
- Predict accident risk using the pre-trained model
- Display the result with feedback and visual cues
- Validate user inputs and handle errors

## 🧑‍💻 Team
- Hana Mohamed Gohar
- - Eman Mousa Kamer 
- Samar Fawzy Abousamra  
- Eman Hussein Batie  
 

**Supervisor:** Dr. Heba El-Hadidi

## 🛠 Tech Stack
- Python, Pandas, Scikit-learn, XGBoost  
- SMOTE, PowerTransformer  
- Matplotlib, Seaborn  
- Tkinter (for GUI)

## 📂 How to Run
1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run `main.py` or the Jupyter Notebook  
4. Launch the GUI to interact with the model  

---


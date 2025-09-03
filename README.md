# 🛡️ Credit Card Fraud Detection

**Author:** Rahul Rai  
**Project Type:** End-to-End Machine Learning & Deep Learning Project  
**Goal:** Detect fraudulent credit card transactions using supervised and unsupervised learning techniques.

---

## **📌 Project Overview**
This project aims to **detect fraudulent credit card transactions** using multiple models:
- **Logistic Regression** → Baseline model
- **XGBoost Classifier** → High-performance supervised model
- **Autoencoder (Deep Learning)** → Unsupervised anomaly detection

The project handles **class imbalance**, evaluates **model performance** using ROC-AUC, PR curves, and confusion matrices, and includes an **optional FastAPI microservice** for real-time fraud detection.

---

## **📂 Project Structure**
Credit-Card-Fraud-Detection/
│── artifacts/
│ ├── model_logreg.pkl # Logistic Regression model
│ ├── model_xgb.pkl # XGBoost model
│ ├── autoencoder.h5 # Autoencoder model
│ ├── scaler.pkl # StandardScaler for preprocessing
│ ├── ae_threshold.json # Autoencoder threshold configuration
│ ├── metrics.json # Stored metrics
│ └── plots/ # Saved plots (PR curves, confusion matrices)
│
│── Credit_Card_Fraud_Detection_Rahul.ipynb # Main notebook
│── app.py # FastAPI app (optional)
│── requirements.txt # Project dependencies
│── README.md # Project documentation
│── creditcard.csv # Dataset (not included)


---

## **⚡ Features**
✅ Handles highly **imbalanced datasets** using **SMOTE**  
✅ Implements **supervised** & **unsupervised** fraud detection  
✅ Saves models, metrics & thresholds automatically  
✅ Generates **precision-recall curves** and **confusion matrices**  
✅ Includes **real-time fraud detection API** using **FastAPI**  
✅ Ready for **deployment**  

---

## **🚀 Installation & Setup**

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download dataset

Download the dataset from Kaggle - Credit Card Fraud Detection

and place it as creditcard.csv in the project root.

🧠 Model Training

Open the notebook:

jupyter notebook Credit_Card_Fraud_Detection_Rahul.ipynb


Or run in Google Colab by uploading the notebook.

📊 Model Performance
Model	ROC-AUC	Precision (Fraud)	Recall (Fraud)	F1-Score
Logistic Regression	0.9808	0.06	0.92	0.11
XGBoost	1.0000	0.81	1.00	0.89
Autoencoder	0.9508	0.12	0.82	0.21

📌 Best Model: XGBoost → Best ROC-AUC & highest fraud detection accuracy.

🌐 FastAPI Deployment (Optional)

Run the API locally:

uvicorn app:app --reload


Then open http://127.0.0.1:8000/docs
 to test the endpoints.

📈 Visualizations

The following plots are saved in artifacts/plots/:

Precision-Recall Curves

Confusion Matrices

Transaction Amount Distributions

Class Imbalance Plot

🛠️ Tech Stack

Languages → Python 3.10+

Libraries → NumPy, Pandas, Scikit-learn, XGBoost, TensorFlow, Matplotlib, Seaborn

API → FastAPI + Uvicorn

Deployment → Local / Colab / Cloud

📌 Author

Rahul Rai
📧 Email: rahulraimau5@gmail.com

💼 LinkedIn: [[linkedin.com/in/rahulrai](https://www.linkedin.com/in/rahul-rai-629554245/)

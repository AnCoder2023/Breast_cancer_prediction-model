Breast Cancer Prediction – Machine Learning Project

This project builds a machine learning model that predicts whether a tumor is Malignant (cancerous) or Benign (non-cancerous) based on medical diagnostic features.
It uses the popular Breast Cancer Wisconsin dataset from Scikit-Learn.

The goal is to help in early diagnosis using supervised machine learning.

📂 Project Structure
breast-cancer-prediction/
 ├── models/
 │     └── breast_cancer_model.pkl
 ├── notebooks/
 │     └── breast_cancer_prediction.ipynb
 ├── README.md

🧬 About the Dataset

The Breast Cancer Wisconsin dataset contains:

569 samples

30 numeric features, including:

Radius

Texture

Smoothness

Compactness

Symmetry

Concavity

Each row describes characteristics of a cell nucleus from a digital breast mass image.

Target classes:

0 → Malignant (cancerous)

1 → Benign (non-cancerous)

🧠 Machine Learning Pipeline
1️⃣ Load and Explore Dataset

Dataset loaded using Scikit-Learn’s load_breast_cancer().

2️⃣ Preprocessing

Split into training and testing sets

Features scaled using StandardScaler

3️⃣ Model Training

A Logistic Regression model was used because it performs well on binary classification problems.

4️⃣ Evaluation Metrics

The model achieved high performance using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1 Score)

5️⃣ Model Saving

The trained model was exported as:

models/breast_cancer_model.pkl

🚀 How to Run the Project

Clone the repo:

git clone <your-repo-link>


Open the notebook:

notebooks/breast_cancer_prediction.ipynb


Run all cells to:

Train the model

Evaluate it

Save predictions

📌 Results

The logistic regression model correctly classified most tumors with high accuracy, making it reliable for early-stage breast cancer prediction.

👩‍💻 Author

Ananya
Machine Learning Intern – CodeAlpha

# Training-ML 🚀

![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)   ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)  ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)   ![Status](https://img.shields.io/badge/status-active-success)  ![Last Commit](https://img.shields.io/github/last-commit/KABUYA-SINA/Machine_Learning)
![Repo Size](https://img.shields.io/github/repo-size/KABUYA-SINA/Machine_Learning)

## 🎯 Objective

This project implements a complete Machine Learning pipeline for binary classification (income prediction).

It demonstrates how to:

* preprocess data
* train a model
* evaluate performance
* visualize results
* save and reuse a model
* expose predictions via a REST API

---

## ⚙️ Tech Stack

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib
* Seaborn
* Flask

---

## 📊 Project Overview

This project follows a standard Machine Learning workflow:

1. Load dataset
2. Clean and preprocess data
3. Train a model (Random Forest)
4. Evaluate performance
5. Visualize results
6. Save trained model (`model.pkl`)
7. Serve predictions via API

---

## 📁 Project Structure

```
training/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── metrics.py
│   ├── visualization.py
│
├── models/
│   └── model.pkl
│
├── flask/
│   └── api.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Training Pipeline

```bash
python app.py
```

This will:

* load and clean data
* train the model
* evaluate performance
* generate visualizations
* save the model in `/models/model.pkl`

---

## 🌐 Run the API

```bash
python flask/api.py
```

API will run on:

```
http://127.0.0.1:5000
```

---

## 📡 API Usage

### Endpoint

```
GET /
GET /health
GET /visualization
POST /predict
```

---

### Example Request

```json
{
  "features": [1, 50, 35, 50000, 0, 1, 3, 2, 12, 100, 5]
}
```

---

### Example Response

```json
{
  "prediction": 0,
    "probabilities": {
        "0": 0.7916445304104879,
        "1": 0.20835546958951212
    }
}
```

---

### Error Example

```json
{
  "error": "Missing features"
}
```

---

## 📈 Model Output

During training, the model prints:

* Accuracy score
* Confusion matrix
* Classification report

It also displays:

* Confusion matrix visualization

---

## 🧠 Model Details

* Algorithm: Random Forest Classifier
* Task: Binary classification
* Target: `PINCP` (income threshold classification)
* Train/Test split: 80/20
* Fixed random state for reproducibility

---

## 💡 Key Features

* Clean and modular code structure
* Reusable and scalable ML pipeline
* Model persistence with `.pkl`
* API for real-time predictions
* Data visualization included

---

## 🔧 Future Improvements

* Add cross-validation
* Improve feature engineering
* Add model comparison (Logistic Regression, XGBoost)
* Deploy API (Render, AWS, etc.)
* Build a frontend interface

---

## 📌 Notes

* Ensure your dataset contains the `PINCP` column
* Input features must match training data format
* API expects numerical input only

---

## 👨‍💻 Author

Machine Learning beginner project focused on building a complete and structured ML pipeline.

---

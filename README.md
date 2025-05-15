
# 🧁 Bakery Sales Prediction

This project forecasts bakery sales for the next **N** days using machine learning models. It uses historical sales data to train and evaluate two models — Linear Regression and Random Forest Regressor — and predicts future sales based on temporal features.

---

## 📌 Objective

To build a machine learning pipeline that:
- Loads and preprocesses bakery sales data
- Trains and evaluates two predictive models
- Selects the best model
- Forecasts sales for the next `N` days
- Outputs prediction and model performance

---

## 📂 Project Structure

```
├── dataset.csv                  # Input data with 'DATE' and 'SALES'
├── sales_predict.py             # Main prediction script
├── model_comparison.csv         # Model performance table (generated)
├── future_predictions.csv       # Predicted future sales (generated)
├── rf_model.pkl                 # Trained Random Forest model (generated)
├── scaler.pkl                   # Standard Scaler object (generated)
└── README.md                    # Project documentation
```

---

## 🚀 How to Run

Make sure you have Python 3.7+ installed.

### 1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Run the script:
```bash
python sales_predict.py <N_days>
```

📌 Example:
```bash
python sales_predict.py 7
```
This will forecast sales for the next 7 days.

---

## 📊 Output

### ✅ Sample Output:

```
Future Sales Predictions (Random Forest):
        DATE  PREDICTED_SALES
0  2025-05-16               42
1  2025-05-17               37
2  2025-05-18               35
3  2025-05-19               39
...
```

### ✅ Metrics Output:
Stored in `model_comparison.csv`:
| Model                 | R2 Score | MSE      | MAE    |
|----------------------|----------|----------|--------|
| Linear Regression     | 0.78     | 123.45   | 8.91   |
| Random Forest Regressor | 0.89  | 88.76    | 6.12   |

---

## ⚙️ Features Used
- Day of week
- Day of month
- Month
- Weekend indicator
- Days since start

---

## ✅ Model Selection

Random Forest Regressor is selected as the final model due to:
- Lower RMSE and MAE
- Better handling of non-linear patterns in temporal data

---

## 📁 Outputs

| File                     | Description                       |
|--------------------------|-----------------------------------|
| `future_predictions.csv` | Forecasted sales for next N days  |
| `model_comparison.csv`   | Evaluation of both models         |
| `rf_model.pkl`           | Trained Random Forest model       |
| `scaler.pkl`             | StandardScaler object             |

---

## 👩‍💻 Author

- **Name**: [Ramya Karanam]
- **Role**: ML Developer (Assignment Submission)

---

## 📄 License

This project is part of an academic/technical assessment and is not licensed for commercial use.

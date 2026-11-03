# RecoverAI – ML-Powered Payment Recovery System

RecoverAI is a machine learning-based payment recovery system that predicts the chances of recovering failed payments. It helps prioritize transactions and suggests suitable recovery actions based on payment details.

## Features

- Predicts the recovery probability of failed payments.
- Classifies payments into HIGH, MEDIUM, and LOW priority.
- Suggests recovery actions based on failure reasons.
- Provides an interactive Streamlit dashboard.
- Displays payment priority distribution using charts.
- Allows users to enter new transaction details and analyze them.
- Includes transaction filters and search options.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Random Forest
- **Dashboard:** Streamlit
- **Data Visualization:** Altair
- **Model Storage:** Joblib

## Project Structure

```text
RecoverAI/
├── data/
│   └── transactions.csv
├── dashboard.py
├── main.py
├── predict.py
├── train_model.py
├── generate_data.py
├── analyze_data.py
├── recovery_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/SanaMahato05/RecoverAI-Payment-Recovery-Prediction.git
```

2. Open the project folder:

```bash
cd RecoverAI-Payment-Recovery-Prediction
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

5. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run dashboard.py
```

## Dataset

The project uses a randomly generated dataset of 1,000 sample payment transactions. It includes payment amounts, payment methods, failure reasons, previous failures, and recovery status.

## Machine Learning Model

RecoverAI uses a Random Forest classification model to estimate the recovery probability of failed payments and support transaction prioritization.

## Future Improvements

- Improve prediction accuracy with more training data.
- Add more payment recovery strategies.
- Connect the system to a payment gateway API.
- Improve the dashboard with additional analytics.

## Author

**Sana Mahato**  


**GitHub:** https://github.com/SanaMahato05

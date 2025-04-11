# Customer Churn Prediction App

A Streamlit web application for predicting customer churn and analyzing customer behavior in e-commerce.

## Features

- **Churn Prediction**: Predict whether a customer will churn based on their behavior
- **Customer Analysis**: Detailed analysis of customer purchase history and behavior
- **Interactive Visualizations**: Charts and metrics to understand customer patterns

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the app locally:
```bash
streamlit run streamlit_app.py
```

## Deployment

This app is deployed on Streamlit Cloud. You can access it at: [Your Streamlit Cloud URL]

## Project Structure

- `streamlit_app.py`: Main application file
- `requirements.txt`: Python dependencies
- `best_churn_model.pkl`: Trained machine learning model
- `cleaned_ecommerce_data.csv`: Sample customer data

## Technologies Used

- Streamlit
- Scikit-learn
- Pandas
- Plotly
- Joblib 
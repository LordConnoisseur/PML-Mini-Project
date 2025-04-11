import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the model from Hugging Face
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/Lord-Connoisseur/Churn_Prediction/resolve/main/best_churn_model_safe.pkl"
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
    )
    
    # Create a session with retry strategy
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    try:
        # Add a timeout to the request
        response = session.get(model_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Verify the content is not empty
        if not response.content:
            raise ValueError("Empty response received from Hugging Face")
            
        # Try to load the model
        try:
            model = joblib.load(io.BytesIO(response.content))
            print("Model Loaded!")
            return model
        except Exception as e:
            print("Model Not Loaded!")
            st.error(f"Error loading model from bytes: {str(e)}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

model = load_model()

# Generate explanation function
def generate_explanation(data):
    contributing = []

    if int(data['Recency']) > 300:
        contributing.append("High Recency")

    if int(data['Frequency']) <= 2:
        contributing.append("Low Frequency")

    if float(data['Total_Amount']) < 100:
        contributing.append("Low Total Amount")

    if float(data['Return_Rate']) > 0.5:
        contributing.append("High Return Rate")

    if not contributing:
        return "No strong contributing factors detected."

    return "Contributing Factors: " + ", ".join(contributing)

# Main app
def main():
    st.title("Customer Churn Prediction Dashboard")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Customer Analysis"]
    )

    if page == "Home":
        st.header("Welcome to Customer Churn Prediction")
        st.write("""
        This application helps predict customer churn and analyze customer behavior.
        
        - Use the **Prediction** page to predict if a customer will churn
        - Use the **Customer Analysis** page to analyze specific customer data
        """)
        
        # Add some statistics or visualizations here if needed

    elif page == "Prediction":
        st.header("Churn Prediction")
        
        if model is None:
            st.error("Model failed to load. Please try refreshing the page or contact support.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
            
        with col2:
            total_amount = st.number_input("Total Amount Spent", min_value=0.0, value=1000.0)
            unique_categories = st.number_input("Unique Categories Purchased", min_value=1, value=3)
            avg_purchase_value = st.number_input("Average Purchase Value", min_value=0.0, value=200.0)
            return_rate = st.slider("Return Rate", min_value=0.0, max_value=1.0, value=0.1)

        if st.button("Predict Churn"):
            try:
                input_data = {
                    "Gender": gender,
                    "Age": age,
                    "Recency": recency,
                    "Frequency": frequency,
                    "Total_Amount": total_amount,
                    "Unique_Categories": unique_categories,
                    "Avg_Purchase_Value": avg_purchase_value,
                    "Return_Rate": return_rate
                }

                input_df = pd.DataFrame([input_data])
                
                # Ensure the model is loaded before making predictions
                if model is None:
                    st.error("Model is not available. Please try again later.")
                    return
                    
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1]

                prediction_label = "Yes" if prediction == 1 else "No"
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Churn Prediction", prediction_label)
                with col2:
                    st.metric("Probability", f"{proba:.2%}")
                
                # Show explanation
                st.subheader("Explanation")
                explanation = generate_explanation(input_data)
                st.info(explanation)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif page == "Customer Analysis":
        st.header("Customer Analysis")
        
        customer_id = st.text_input("Enter Customer ID")
        
        if customer_id:
            try:
                # Load the dataset
                @st.cache_data
                def load_data():
                    return pd.read_csv("cleaned_ecommerce_data.csv")
                
                df = load_data()
                
                # Calculate Total Purchase if it doesn't exist
                if 'Total Purchase' not in df.columns:
                    df['Total Purchase'] = df['Product Price'] * df['Quantity']
                
                # Convert Customer ID to string if needed
                df['Customer ID'] = df['Customer ID'].astype(str)
                
                # Filter data for the specific customer
                customer_data = df[df['Customer ID'] == customer_id.strip()]
                
                if customer_data.empty:
                    st.error(f"No data found for Customer ID: {customer_id}")
                else:
                    # Calculate metrics
                    metrics = {
                        'total_spent': customer_data['Total Purchase'].sum(),
                        'avg_purchase': customer_data['Total Purchase'].mean(),
                        'total_orders': len(customer_data),
                        'favorite_category': customer_data['Product Category'].mode()[0],
                        'last_purchase': customer_data['Purchase Date'].max(),
                        'return_rate': (customer_data['Returns'].sum() / len(customer_data)) * 100
                    }
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Spent", f"${metrics['total_spent']:,.2f}")
                        st.metric("Average Purchase", f"${metrics['avg_purchase']:,.2f}")
                    with col2:
                        st.metric("Total Orders", metrics['total_orders'])
                        st.metric("Favorite Category", metrics['favorite_category'])
                    with col3:
                        st.metric("Last Purchase", metrics['last_purchase'])
                        st.metric("Return Rate", f"{metrics['return_rate']:.1f}%")
                    
                    # Monthly spending chart
                    monthly_spending = customer_data.groupby('Purchase Month')['Total Purchase'].sum()
                    fig1 = px.line(x=monthly_spending.index, y=monthly_spending.values,
                                 title="Monthly Spending Trend")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Category distribution
                    category_dist = customer_data['Product Category'].value_counts()
                    fig2 = px.pie(values=category_dist.values, names=category_dist.index,
                                title="Purchase Distribution by Category")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display recent purchases
                    st.subheader("Recent Purchases")
                    st.dataframe(customer_data[['Purchase Date', 'Product Category', 'Product Price', 'Quantity', 'Total Purchase']]
                               .sort_values('Purchase Date', ascending=False)
                               .head(10))
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
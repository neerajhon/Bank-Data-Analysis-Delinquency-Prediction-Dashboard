import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set page configuration (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Bank Data Analysis & Prediction Dashboard 🌟", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header { color: #2E86C1; font-size: 36px; text-align: center; }
    .sub-header { color: #3498DB; font-size: 24px; }
    .metric-box { background-color: #E8F6F3; padding: 10px; border-radius: 10px; }
    .stButton>button { background-color: #28B463; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    df_credit = pd.read_csv('credit_card (1).csv')
    df_demographics = pd.read_csv('customer (1).csv')
    df = pd.merge(df_credit, df_demographics, on='Client_Num', how='inner')
    
    # Data cleaning and feature engineering
    df['House_Owner'] = df['House_Owner'].apply(lambda x: 1 if x == "yes" else 0)
    df['Car_Owner'] = df['Car_Owner'].apply(lambda x: 1 if x == "yes" else 0)
    df['Personal_loan'] = df['Personal_loan'].apply(lambda x: 1 if x == "yes" else 0)
    df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'], dayfirst=True)
    
    # Feature engineering
    bins = [18, 25, 35, 45, 55, 65, np.inf]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)
    
    income_bins = [0, 25000, 50000, 75000, 100000, np.inf]
    income_labels = ['<25K', '25K-50K', '50K-75K', '75K-100K', '100K+']
    df['Income_Category'] = pd.cut(df['Income'], bins=income_bins, labels=income_labels, right=False)
    
    df['Net_Revenue_per_Customer'] = df['Interest_Earned'] + df['Annual_Fees'] - df['Customer_Acq_Cost']
    df['Avg_Transaction_Value'] = df['Total_Trans_Amt'] / df['Total_Trans_Vol'].replace(0, 1)
    df['Card_Activation_Date_Proxy'] = df.groupby('Client_Num')['Week_Start_Date'].transform('min')
    df['Card_Age_Days'] = (df['Week_Start_Date'] - df['Card_Activation_Date_Proxy']).dt.days
    df['Month_Year'] = df['Week_Start_Date'].dt.to_period('M').astype(str)
    
    return df

df = load_data()

# --- Sidebar for Page Navigation and Filters ---
st.sidebar.markdown("<h2 style='color: #3498DB;'>Navigation & Filters 🚀</h2>", unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose a Page 📄", ["Visualizations 📊", "Prediction 🔍"])

st.sidebar.markdown("<h3 style='color: #3498DB;'>Data Filters 🎛️</h3>", unsafe_allow_html=True)
card_category = st.sidebar.selectbox("Card Category 💳", options=["All"] + list(df['Card_Category'].unique()), index=0)
age_group = st.sidebar.selectbox("Age Group 👥", options=["All"] + list(df['Age_Group'].unique()), index=0)
income_category = st.sidebar.selectbox("Income Category 💰", options=["All"] + list(df['Income_Category'].unique()), index=0)
delinquency_status = st.sidebar.selectbox("Delinquency Status ⚖️", options=["All", "Not Delinquent", "Delinquent"], index=0)

# Apply filters
filtered_df = df.copy()
if card_category != "All":
    filtered_df = filtered_df[filtered_df['Card_Category'] == card_category]
if age_group != "All":
    filtered_df = filtered_df[filtered_df['Age_Group'] == age_group]
if income_category != "All":
    filtered_df = filtered_df[filtered_df['Income_Category'] == income_category]
if delinquency_status == "Not Delinquent":
    filtered_df = filtered_df[filtered_df['Delinquent_Acc'] == 0]
elif delinquency_status == "Delinquent":
    filtered_df = filtered_df[filtered_df['Delinquent_Acc'] == 1]

# --- Visualizations Page ---
if page == "Visualizations 📊":
    st.markdown("<h1 class='main-header'>Bank Data Visualizations Dashboard 📈</h1>", unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("<h2 class='sub-header'>Key Metrics 📊</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Total Customers 👥", len(filtered_df))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Avg Transaction Amount 💸", f"${filtered_df['Total_Trans_Amt'].mean():,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Avg Credit Limit 💳", f"${filtered_df['Credit_Limit'].mean():,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Delinquency Rate ⚠️", f"{filtered_df['Delinquent_Acc'].mean()*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # EDA Visualizations
    st.markdown("<h2 class='sub-header'>Exploratory Data Analysis 🔍</h2>", unsafe_allow_html=True)

    # Customer Age Distribution
    st.write("Customer Age Distribution 👥")
    fig1 = px.histogram(filtered_df, x="Customer_Age", nbins=30, title="Customer Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Credit Limit by Card Category
    st.write("Credit Limit by Card Category 💳")
    fig2 = px.box(filtered_df, x="Card_Category", y="Credit_Limit", title="Credit Limit by Card Category")
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly Spending Trends
    st.write("Monthly Total Transaction Amount Trend 📅")
    monthly_spending = filtered_df.groupby('Month_Year')['Total_Trans_Amt'].sum().reset_index()
    fig3 = px.line(monthly_spending, x='Month_Year', y='Total_Trans_Amt', title="Monthly Transaction Amount Trend")
    st.plotly_chart(fig3, use_container_width=True)

    # Income vs. Total Transaction Amount
    st.write("Income vs. Total Transaction Amount 💰")
    fig4 = px.scatter(filtered_df, x="Income", y="Total_Trans_Amt", color="Delinquent_Acc",
                      title="Income vs. Total Transaction Amount (by Delinquency)",
                      labels={"Delinquent_Acc": "Delinquency Status"})
    st.plotly_chart(fig4, use_container_width=True)

    # Correlation Heatmap (Improved for Clarity)
    st.write("Correlation Heatmap 📉")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    correlation_matrix = filtered_df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))  # Increased figure size for better readability
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8},  # Smaller font size
                cbar_kws={'shrink': 0.8}, linewidths=0.5, square=True)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for clarity
    plt.yticks(rotation=0)  # Keep y-axis labels vertical
    plt.title("Correlation Matrix", pad=20)
    st.pyplot(plt)

    # Model Performance Metrics
    st.markdown("<h2 class='sub-header'>Model Performance Metrics 🏆</h2>", unsafe_allow_html=True)
    st.write("XGBoost Model (Class Weights) 🚀")
    st.write("✅ Accuracy: 0.9042")
    st.write("✅ Precision: 0.0680")
    st.write("✅ Recall: 0.0458")
    st.write("✅ F1-Score: 0.0547")
    st.write("✅ ROC AUC Score: 0.5429")

    # Download Filtered Data
    st.markdown("<h2 class='sub-header'>Download Filtered Data 📥</h2>", unsafe_allow_html=True)
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV 📄", csv, "filtered_bank_data.csv", "text/csv")

# --- Prediction Page ---
elif page == "Prediction 🔍":
    st.markdown("<h1 class='main-header'>Delinquency Prediction Dashboard 🔮</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Predict Delinquency 📝</h2>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.write("Enter Customer Details for Delinquency Prediction ✍️")
        col1, col2 = st.columns(2)
        
        with col1:
            customer_age = st.number_input("Customer Age 👥", min_value=18, max_value=100, value=30)
            income = st.number_input("Income ($) 💰", min_value=0, value=50000)
            credit_limit = st.number_input("Credit Limit ($) 💳", min_value=0.0, value=5000.0)
            total_trans_amt = st.number_input("Total Transaction Amount ($) 💸", min_value=0, value=4000)
            total_trans_vol = st.number_input("Total Transaction Volume 🔢", min_value=0, value=50)
            card_category = st.selectbox("Card Category 💳", options=df['Card_Category'].unique())
            gender = st.selectbox("Gender 🚻", options=df['Gender'].unique())
            education_level = st.selectbox("Education Level 🎓", options=df['Education_Level'].unique())
        
        with col2:
            marital_status = st.selectbox("Marital Status 💍", options=df['Marital_Status'].unique())
            use_chip = st.selectbox("Use Chip 💳", options=df['Use Chip'].unique())
            exp_type = st.selectbox("Expense Type 🛍️", options=df['Exp Type'].unique())
            annual_fees = st.number_input("Annual Fees ($) 💵", min_value=0, value=200)
            customer_acq_cost = st.number_input("Customer Acquisition Cost ($) 📈", min_value=0, value=100)
            interest_earned = st.number_input("Interest Earned ($) 💹", min_value=0.0, value=500.0)
            cust_satisfaction_score = st.number_input("Customer Satisfaction Score 😊", min_value=1, max_value=5, value=3)
            dependent_count = st.number_input("Dependent Count 👨‍👩‍👧", min_value=0, value=2)
        
        submitted = st.form_submit_button("Predict Delinquency 🔍")
        
        if submitted:
            # Load the trained model
            model = joblib.load('delinquency_prediction_model_updated.pkl')  # Change to 'delinquency_prediction_model.pkl' if not resaved
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Customer_Age': [customer_age],
                'Income': [income],
                'Credit_Limit': [credit_limit],
                'Total_Trans_Amt': [total_trans_amt],
                'Total_Trans_Vol': [total_trans_vol],
                'Card_Category': [card_category],
                'Gender': [gender],
                'Education_Level': [education_level],
                'Marital_Status': [marital_status],
                'Use Chip': [use_chip],
                'Exp Type': [exp_type],
                'Annual_Fees': [annual_fees],
                'Customer_Acq_Cost': [customer_acq_cost],
                'Interest_Earned': [interest_earned],
                'Cust_Satisfaction_Score': [cust_satisfaction_score],
                'Dependent_Count': [dependent_count],
                'Activation_30_Days': [1],  # Default value
                'current_year': [2023],  # Constant from dataset
                'Total_Revolving_Bal': [0],  # Default
                'Avg_Utilization_Ratio': [0.0],  # Default
                'Week_Num': ['Week-1'],  # Default
                'Qtr': ['Q1'],  # Default
                'state_cd': ['CA'],  # Default
                'Zipcode': [91750],  # Default
                'Car_Owner': [0],  # Default
                'House_Owner': [0],  # Default
                'Personal_loan': [0],  # Default
                'contact': ['cellular'],  # Default
                'Customer_Job': ['Selfemployeed'],  # Default
                'Age_Group': [pd.cut([customer_age], bins=[18, 25, 35, 45, 55, 65, np.inf], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], right=False)[0]],
                'Income_Category': [pd.cut([income], bins=[0, 25000, 50000, 75000, 100000, np.inf], labels=['<25K', '25K-50K', '50K-75K', '75K-100K', '100K+'], right=False)[0]],
                'Net_Revenue_per_Customer': [interest_earned + annual_fees - customer_acq_cost],
                'Avg_Transaction_Value': [total_trans_amt / max(total_trans_vol, 1)],  # Correct column name
                'Card_Age_Days': [0]  # Default
            })
            
            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.markdown("<h3 class='sub-header'>Prediction Result 🎯</h3>", unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"Customer is predicted to be **Delinquent** 😟 (Probability: {probability*100:.2f}%)")
            else:
                st.success(f"Customer is predicted to be **Not Delinquent** ✅ (Probability of Delinquency: {probability*100:.2f}%)")
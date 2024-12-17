import streamlit as st
import pandas as pd
from huggingface_hub import InferenceApi
import plotly.express as px

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(page_title="AI-Powered Policy Impact Analyzer", page_icon="🤖", layout="wide")

# --- Hugging Face API Setup ---
api = InferenceApi(repo_id="distilbert-base-cased-distilled-squad")

# Function to Query Hugging Face API
def query_hf_api(question, context):
    payload = {"inputs": {"question": question, "context": context}}
    result = api(payload)
    return result.get("answer", "No answer found.")

# --- Load Data from Local CSV File ---
@st.cache_data
def load_data():
    file_path = "Combined_Financial_Data.csv"  # CSV file stored in the same folder as app.py
    data = pd.read_csv(file_path)
    context = "\n".join(data.apply(lambda row: f"{row['Category']}: {row['Amount']}", axis=1))
    return data, context

# Load Data
data, context = load_data()

# --- Streamlit App Layout ---
st.title("🤖 AI-Powered Policy Impact Analyzer")
st.write("""
Welcome to the **AI-Powered Policy Impact Analyzer**.  
Ask any question about **expenditures, taxes, or policies**, and our AI will provide accurate answers based on the data.
""")
st.markdown("---")

# Data Preview Section
st.subheader("📋 Reference Data")
st.dataframe(data)

# Visualization Section
st.subheader("📊 Visualize the Data")
chart_type = st.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
x_axis = st.selectbox("Select X-axis", data.columns)
y_axis = st.selectbox("Select Y-axis", data.columns)

if chart_type == "Bar Chart":
    fig = px.bar(data, x=x_axis, y=y_axis, title="Bar Chart")
    st.plotly_chart(fig)
elif chart_type == "Line Chart":
    fig = px.line(data, x=x_axis, y=y_axis, title="Line Chart")
    st.plotly_chart(fig)
elif chart_type == "Pie Chart":
    fig = px.pie(data, names=x_axis, values=y_axis, title="Pie Chart")
    st.plotly_chart(fig)

# User Query Section
st.subheader("🔍 Ask Your Policy Question")
user_question = st.text_input("Enter your question here:", placeholder="e.g., 'What is the expenditure on Defence?'")

if user_question:
    st.info("🔄 Analyzing your question...")
    try:
        # Query Hugging Face API
        answer = query_hf_api(user_question, context)
        
        if answer != "No answer found.":
            st.success(f"**Answer:** {answer}")
        else:
            st.warning("AI could not find a relevant answer. Please try rephrasing your question.")
    except Exception as e:
        st.error("An error occurred while processing your request.")
        st.write(str(e))

# Footer
st.markdown("---")
st.write("**Developed with ❤️ using Streamlit, Pandas, Plotly, and Hugging Face API**")

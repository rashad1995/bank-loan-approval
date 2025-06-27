import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------ Setup Page ------------------------
st.set_page_config(
    page_title="Smart Loan Approval",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ Custom Style ------------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    h1, h2, h3 {
        color: #0f4c81;
    }
    .stButton > button {
        font-size: 18px;
        background-color: #0f4c81;
        color: white;
        border-radius: 8px;
        padding: 0.4em 1em;
    }
    .stSelectbox, .stNumberInput {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ Load Model ------------------------
@st.cache_resource
def load_model():
    with open("endpoint_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------------ Initialize Session ------------------------
if "requests" not in st.session_state:
    st.session_state.requests = pd.DataFrame(columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Prediction'
    ])

# ------------------------ Page 1: Add Loan ------------------------
def add_request_page():
    st.title("📝 Submit a New Loan Application")

    st.markdown("#### 📌 Fill in applicant details:")

    with st.form("loan_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("👤 Gender", ["Male", "Female"])
            married = st.selectbox("💍 Married", ["Yes", "No"])
            dependents = st.selectbox("👶 Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
            property_area = st.selectbox("🌍 Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income = st.number_input("💰 Applicant Income", 0)
            coapplicant_income = st.number_input("💵 Coapplicant Income", 0)
            loan_amount = st.number_input("🏦 Loan Amount", 0)
            loan_term = st.selectbox("📆 Loan Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360])
            credit_history = st.selectbox("🔐 Credit History", [1.0, 0.0])

        submitted = st.form_submit_button("✅ Predict & Submit")

        if submitted:
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }])

            prediction = model.predict(input_df)[0]
            input_df["Prediction"] = prediction

            st.session_state.requests = pd.concat([st.session_state.requests, input_df], ignore_index=True)

            if prediction == "Y":
                st.success("✅ Approved! The loan is likely to be granted.")
            else:
                st.error("❌ Rejected. The loan is likely to be declined.")

    # ---- Delete Requests Section ----
    st.divider()
    st.subheader("🗑️ Manage Submitted Requests")

    df = st.session_state.requests
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        delete_index = st.number_input("Enter row index to delete", 0, len(df) - 1)
        if st.button("❌ Delete Selected Request"):
            st.session_state.requests = df.drop(index=delete_index).reset_index(drop=True)
            st.warning("Deleted successfully.")
    else:
        st.info("No submitted loan requests yet.")

# ------------------------ Page 2: EDA ------------------------
def eda_page():
    st.title("📊 Exploratory Data Analysis")

    df = st.session_state.requests
    if df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("📈 Approval Status Distribution")
    st.bar_chart(df["Prediction"].value_counts())

    st.subheader("💰 Income vs Loan Amount")
    st.scatter_chart(df[["ApplicantIncome", "LoanAmount"]])

# ------------------------ Page 3: Data Issues ------------------------
def data_report_page():
    st.title("🧹 Data Quality Report")

    df = st.session_state.requests
    if df.empty:
        st.info("No data available.")
        return

    st.subheader("📋 Missing Value Check")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count"}))

# ------------------------ Page 4: Model Metrics ------------------------
def metrics_page():
    st.title("📈 Model Performance")

    df = st.session_state.requests
    if df.empty or "Actual" not in df.columns:
        st.warning("Please add an 'Actual' column to compute evaluation metrics.")
        return

    y_true = df["Actual"]
    y_pred = df["Prediction"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, pos_label='Y'):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, pos_label='Y'):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred, pos_label='Y'):.2f}")

# ------------------------ Sidebar Navigation ------------------------
st.sidebar.title("📌 Navigation")
choice = st.sidebar.radio("Go to", ["🏠 Submit Request", "📊 EDA", "🧹 Data Report", "📈 Model Metrics"])

if choice == "🏠 Submit Request":
    add_request_page()
elif choice == "📊 EDA":
    eda_page()
elif choice == "🧹 Data Report":
    data_report_page()
elif choice == "📈 Model Metrics":
    metrics_page()

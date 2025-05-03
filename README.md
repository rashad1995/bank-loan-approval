# Loan Approval Web Application using Machine Learning

This project is a web-based application that helps banks automate the decision-making process for loan approvals or rejections. It uses a machine learning model trained on real data to predict whether an individual loan request should be approved. The goal is to provide a complete pipeline including data analysis, model training, performance evaluation, and an interactive user interface.

## Project Features

- Add/Delete individual loan requests
- Display predictions (approve/reject)
- Exploratory Data Analysis (EDA) of all requests
- Machine learning model with performance metrics (Accuracy, Precision, Recall, F1-score)
- A dedicated page for date-related issues and how they are handled
- Git branches for teamwork and version control
- Deployed website (link to be added in the report)

## How to Run the Project Locally

1. **Clone the repository:**

```bash
git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name
```

2. **(Optional) Create and activate a virtual environment:**

```bash
python -m venv env
source env/bin/activate      # On Linux/macOS
env\Scripts\activate         # On Windows
```

3. **Install required libraries:**

```bash
pip install -r requirements.txt
```

4. **Run the application:**

- For **Streamlit**:
```bash
streamlit run app.py
```
- For **Flask**:
```bash
python app.py
```

5. **Open the app in your browser:**

- For **Streamlit**: `http://localhost:8501`  
- For **Flask**: `http://127.0.0.1:5000`

---

## Python Requirements

To ensure the project runs properly, you need to install the required libraries listed in the `requirements.txt` file.

### Contents of `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
joblib
```

### Quick Explanation of Libraries:

| Library         | Purpose                                      |
|-----------------|----------------------------------------------|
| `pandas`        | Data manipulation and analysis               |
| `numpy`         | Numerical operations and array handling      |
| `scikit-learn`  | Building and evaluating machine learning models |
| `matplotlib`    | Basic plotting and visualizations            |
| `seaborn`       | Advanced data visualization and plotting     |
| `plotly`        | Interactive plotting (optional)              |
| `streamlit`     | Build interactive web apps quickly           |
| `joblib`        | Saving and loading trained models            |

---
## Team Members

- Rashad Khurma – `rashad_306889`  
- Ali Abbas – `ali_241692`  
- Reem Ahmad – `reem_278333`


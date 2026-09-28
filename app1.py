# ============================================================
# LOAN APPROVAL ANALYTICS
# Complete Data Analytics + Machine Learning Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💰 Loan Approval Analytics")

st.write(
    """
    **End-to-End Data Analytics and Machine Learning Project**

    This application analyzes loan applications and identifies
    the factors associated with loan approval.
    """
)

st.divider()


# ============================================================
# CREATE SAMPLE DATASET
# ============================================================

@st.cache_data
def create_dataset():

    np.random.seed(42)

    n = 1000

    data = pd.DataFrame({

        "Loan_ID":
            ["LN" + str(i).zfill(4) for i in range(1, n + 1)],

        "Gender":
            np.random.choice(
                ["Male", "Female"],
                n,
                p=[0.75, 0.25]
            ),

        "Married":
            np.random.choice(
                ["Yes", "No"],
                n,
                p=[0.65, 0.35]
            ),

        "Dependents":
            np.random.choice(
                ["0", "1", "2", "3+"],
                n,
                p=[0.55, 0.20, 0.15, 0.10]
            ),

        "Education":
            np.random.choice(
                ["Graduate", "Not Graduate"],
                n,
                p=[0.70, 0.30]
            ),

        "Self_Employed":
            np.random.choice(
                ["Yes", "No"],
                n,
                p=[0.15, 0.85]
            ),

        "ApplicantIncome":
            np.random.randint(
                2000,
                15000,
                n
            ),

        "CoapplicantIncome":
            np.random.randint(
                0,
                8000,
                n
            ),

        "LoanAmount":
            np.random.randint(
                50,
                500,
                n
            ),

        "Loan_Amount_Term":
            np.random.choice(
                [120, 180, 240, 300, 360, 480],
                n
            ),

        "Credit_History":
            np.random.choice(
                [0, 1],
                n,
                p=[0.18, 0.82]
            ),

        "Property_Area":
            np.random.choice(
                ["Urban", "Semiurban", "Rural"],
                n,
                p=[0.35, 0.40, 0.25]
            )
    })

    # --------------------------------------------------------
    # Create realistic approval probability
    # --------------------------------------------------------

    score = (
        0.35 * data["Credit_History"]
        + 0.15 * (data["Education"] == "Graduate").astype(int)
        + 0.10 * (data["ApplicantIncome"] > 5000).astype(int)
        + 0.10 * (data["LoanAmount"] < 300).astype(int)
        + 0.10 * (data["Property_Area"] == "Semiurban").astype(int)
        + 0.05 * (data["Married"] == "Yes").astype(int)
        + 0.05 * (data["Self_Employed"] == "No").astype(int)
        + np.random.normal(0, 0.10, n)
    )

    probability = 1 / (1 + np.exp(-6 * (score - 0.55)))

    data["Loan_Status"] = np.where(
        np.random.random(n) < probability,
        "Approved",
        "Rejected"
    )

    # --------------------------------------------------------
    # Add a few missing values
    # --------------------------------------------------------

    data.loc[
        np.random.choice(n, 20, replace=False),
        "Self_Employed"
    ] = np.nan

    data.loc[
        np.random.choice(n, 15, replace=False),
        "Credit_History"
    ] = np.nan

    data.loc[
        np.random.choice(n, 15, replace=False),
        "LoanAmount"
    ] = np.nan

    return data


df = create_dataset()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Data Exploration",
        "Applicant Analysis",
        "Loan Analysis",
        "SQL Analytics",
        "Machine Learning",
        "Loan Prediction",
        "Business Insights"
    ]
)


# ============================================================
# DATA CLEANING
# ============================================================

clean_df = df.copy()

# Numerical columns
numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Categorical columns
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

# Fill numerical missing values
for column in numerical_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )

# Fill categorical missing values
for column in categorical_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].mode()[0]
    )


# ============================================================
# CALCULATED COLUMNS
# ============================================================

clean_df["TotalIncome"] = (
    clean_df["ApplicantIncome"]
    + clean_df["CoapplicantIncome"]
)

clean_df["LoanIncomeRatio"] = (
    clean_df["LoanAmount"]
    / clean_df["TotalIncome"]
)

clean_df["LoanIncomeRatio"] = (
    clean_df["LoanIncomeRatio"].replace(
        [np.inf, -np.inf],
        np.nan
    )
)

clean_df["LoanIncomeRatio"] = (
    clean_df["LoanIncomeRatio"].fillna(0)
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Loan Approval Dashboard")

    total_applications = len(clean_df)

    approved = (
        clean_df["Loan_Status"] == "Approved"
    ).sum()

    rejected = (
        clean_df["Loan_Status"] == "Rejected"
    ).sum()

    approval_rate = (
        approved / total_applications * 100
    )

    average_income = (
        clean_df["TotalIncome"].mean()
    )

    average_loan = (
        clean_df["LoanAmount"].mean()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Applications",
        total_applications
    )

    col2.metric(
        "Approved",
        approved
    )

    col3.metric(
        "Rejected",
        rejected
    )

    col4.metric(
        "Approval Rate",
        f"{approval_rate:.2f}%"
    )

    col5.metric(
        "Avg Loan Amount",
        f"{average_loan:.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # APPROVAL DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Loan Approval Distribution")

        status_counts = (
            clean_df["Loan_Status"]
            .value_counts()
        )

        fig, ax = plt.subplots()

        ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Approved vs Rejected Applications"
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Applications by Property Area")

        area_counts = (
            clean_df["Property_Area"]
            .value_counts()
        )

        fig, ax = plt.subplots()

        ax.bar(
            area_counts.index,
            area_counts.values
        )

        ax.set_xlabel("Property Area")
        ax.set_ylabel("Applications")

        st.pyplot(fig)

    # --------------------------------------------------------
    # CREDIT HISTORY
    # --------------------------------------------------------

    st.subheader(
        "Credit History vs Loan Approval"
    )

    credit_analysis = pd.crosstab(
        clean_df["Credit_History"],
        clean_df["Loan_Status"]
    )

    st.dataframe(credit_analysis)


# ============================================================
# DATA EXPLORATION
# ============================================================

elif page == "Data Exploration":

    st.header("🔎 Data Exploration")

    st.subheader("Dataset Preview")

    st.dataframe(
        clean_df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        clean_df.shape[0]
    )

    col2.metric(
        "Columns",
        clean_df.shape[1]
    )

    st.subheader("Data Types")

    st.dataframe(
        clean_df.dtypes.astype(str)
    )

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(missing)

    st.subheader("Statistical Summary")

    st.dataframe(
        clean_df.describe()
    )

    st.subheader("Duplicate Records")

    duplicates = clean_df.duplicated().sum()

    st.write(
        f"Number of duplicate records: **{duplicates}**"
    )


# ============================================================
# APPLICANT ANALYSIS
# ============================================================

elif page == "Applicant Analysis":

    st.header("👤 Applicant Analysis")

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    st.subheader(
        "Education vs Loan Approval"
    )

    education_analysis = pd.crosstab(
        clean_df["Education"],
        clean_df["Loan_Status"]
    )

    st.dataframe(
        education_analysis
    )

    fig, ax = plt.subplots()

    education_analysis.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Education")
    ax.set_ylabel("Number of Applicants")
    ax.set_title(
        "Education vs Loan Approval"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # SELF EMPLOYED
    # --------------------------------------------------------

    st.subheader(
        "Employment Status vs Loan Approval"
    )

    employment_analysis = pd.crosstab(
        clean_df["Self_Employed"],
        clean_df["Loan_Status"]
    )

    st.dataframe(
        employment_analysis
    )

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    st.subheader(
        "Income Distribution"
    )

    fig, ax = plt.subplots()

    ax.hist(
        clean_df["TotalIncome"],
        bins=30
    )

    ax.set_xlabel("Total Income")
    ax.set_ylabel("Number of Applicants")

    st.pyplot(fig)

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    st.subheader(
        "Gender vs Loan Approval"
    )

    gender_analysis = pd.crosstab(
        clean_df["Gender"],
        clean_df["Loan_Status"]
    )

    st.dataframe(
        gender_analysis
    )


# ============================================================
# LOAN ANALYSIS
# ============================================================

elif page == "Loan Analysis":

    st.header("🏦 Loan Analysis")

    # --------------------------------------------------------
    # LOAN AMOUNT
    # --------------------------------------------------------

    st.subheader(
        "Loan Amount Distribution"
    )

    fig, ax = plt.subplots()

    ax.hist(
        clean_df["LoanAmount"],
        bins=30
    )

    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Applicants")

    st.pyplot(fig)

    # --------------------------------------------------------
    # INCOME VS LOAN AMOUNT
    # --------------------------------------------------------

    st.subheader(
        "Applicant Income vs Loan Amount"
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=clean_df,
        x="TotalIncome",
        y="LoanAmount",
        hue="Loan_Status",
        ax=ax
    )

    ax.set_title(
        "Income vs Loan Amount"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # PROPERTY AREA
    # --------------------------------------------------------

    st.subheader(
        "Property Area vs Approval"
    )

    property_analysis = pd.crosstab(
        clean_df["Property_Area"],
        clean_df["Loan_Status"]
    )

    st.dataframe(
        property_analysis
    )

    # --------------------------------------------------------
    # LOAN TERM
    # --------------------------------------------------------

    st.subheader(
        "Loan Term Analysis"
    )

    term_analysis = pd.crosstab(
        clean_df["Loan_Amount_Term"],
        clean_df["Loan_Status"]
    )

    st.dataframe(
        term_analysis
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "Numerical Feature Correlation"
    )

    numerical_df = clean_df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        ax=ax
    )

    st.pyplot(fig)


# ============================================================
# SQL ANALYTICS
# ============================================================

elif page == "SQL Analytics":

    st.header("🗄️ SQL Analytics")

    st.write(
        """
        The following analyses represent the SQL queries
        that can be implemented when this dataset is stored
        in MySQL, PostgreSQL or SQL Server.
        """
    )

    # --------------------------------------------------------
    # QUERY 1
    # --------------------------------------------------------

    st.subheader(
        "1. Total Number of Applications"
    )

    st.code(
        """
SELECT COUNT(*) AS total_applications
FROM loan_data;
        """,
        language="sql"
    )

    st.write(
        f"Result: **{len(clean_df)} applications**"
    )

    # --------------------------------------------------------
    # QUERY 2
    # --------------------------------------------------------

    st.subheader(
        "2. Approved and Rejected Applications"
    )

    st.code(
        """
SELECT
    Loan_Status,
    COUNT(*) AS total_applications
FROM loan_data
GROUP BY Loan_Status;
        """,
        language="sql"
    )

    status_result = (
        clean_df["Loan_Status"]
        .value_counts()
        .reset_index()
    )

    status_result.columns = [
        "Loan_Status",
        "Total_Applications"
    ]

    st.dataframe(status_result)

    # --------------------------------------------------------
    # QUERY 3
    # --------------------------------------------------------

    st.subheader(
        "3. Approval Rate by Education"
    )

    st.code(
        """
SELECT
    Education,
    COUNT(*) AS total_applications,
    SUM(
        CASE
            WHEN Loan_Status = 'Approved'
            THEN 1 ELSE 0
        END
    ) AS approved,
    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN Loan_Status = 'Approved'
                THEN 1 ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS approval_rate
FROM loan_data
GROUP BY Education;
        """,
        language="sql"
    )

    education_result = (
        clean_df
        .groupby("Education")
        .agg(
            total_applications=(
                "Loan_ID",
                "count"
            ),
            approved=(
                "Loan_Status",
                lambda x:
                (x == "Approved").sum()
            )
        )
    )

    education_result["approval_rate"] = (
        education_result["approved"]
        / education_result["total_applications"]
        * 100
    )

    st.dataframe(
        education_result.reset_index()
    )

    # --------------------------------------------------------
    # QUERY 4
    # --------------------------------------------------------

    st.subheader(
        "4. Approval Rate by Property Area"
    )

    st.code(
        """
SELECT
    Property_Area,
    COUNT(*) AS total_applications,
    SUM(
        CASE
            WHEN Loan_Status = 'Approved'
            THEN 1 ELSE 0
        END
    ) AS approved
FROM loan_data
GROUP BY Property_Area;
        """,
        language="sql"
    )

    property_result = (
        clean_df
        .groupby("Property_Area")
        .agg(
            total_applications=(
                "Loan_ID",
                "count"
            ),
            approved=(
                "Loan_Status",
                lambda x:
                (x == "Approved").sum()
            )
        )
    )

    property_result["approval_rate"] = (
        property_result["approved"]
        / property_result["total_applications"]
        * 100
    )

    st.dataframe(
        property_result.reset_index()
    )

    # --------------------------------------------------------
    # QUERY 5
    # --------------------------------------------------------

    st.subheader(
        "5. Average Income by Loan Status"
    )

    st.code(
        """
SELECT
    Loan_Status,
    AVG(ApplicantIncome) AS average_income
FROM loan_data
GROUP BY Loan_Status;
        """,
        language="sql"
    )

    income_result = (
        clean_df
        .groupby("Loan_Status")[
            "ApplicantIncome"
        ]
        .mean()
        .reset_index()
    )

    income_result.columns = [
        "Loan_Status",
        "Average_Income"
    ]

    st.dataframe(
        income_result
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning")

    st.write(
        """
        We train two classification models to predict
        whether a loan application is likely to be approved.
        """
    )

    ml_df = clean_df.copy()

    # Remove ID
    ml_df = ml_df.drop(
        columns=["Loan_ID"]
    )

    # Convert target
    ml_df["Loan_Status"] = (
        ml_df["Loan_Status"]
        .map({
            "Approved": 1,
            "Rejected": 0
        })
    )

    X = ml_df.drop(
        columns=["Loan_Status"]
    )

    y = ml_df["Loan_Status"]

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                LabelEncoder()
            )
        ]
    )

    # --------------------------------------------------------
    # Custom preprocessing
    # --------------------------------------------------------

    X_processed = X.copy()

    for column in categorical_features:

        X_processed[column] = (
            X_processed[column]
            .astype("category")
            .cat.codes
        )

    X_processed = X_processed.fillna(
        X_processed.median(
            numeric_only=True
        )
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_processed,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_prediction = (
        logistic_model.predict(X_test)
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_prediction = (
        random_forest.predict(X_test)
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_prediction
    )

    logistic_precision = precision_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )

    logistic_recall = recall_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )

    logistic_f1 = f1_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )

    rf_accuracy = accuracy_score(
        y_test,
        rf_prediction
    )

    rf_precision = precision_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    results = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Logistic Regression": [
            logistic_accuracy,
            logistic_precision,
            logistic_recall,
            logistic_f1
        ],

        "Random Forest": [
            rf_accuracy,
            rf_precision,
            rf_recall,
            rf_f1
        ]
    })

    st.subheader(
        "Model Performance Comparison"
    )

    st.dataframe(
        results.style.format(
            {
                "Logistic Regression":
                    "{:.2%}",
                "Random Forest":
                    "{:.2%}"
            }
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance — Random Forest"
    )

    importance = pd.DataFrame({

        "Feature":
            X_processed.columns,

        "Importance":
            random_forest.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    st.dataframe(
        importance
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=importance.head(10),
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "Top 10 Important Features"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Random Forest Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        rf_prediction
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)


# ============================================================
# LOAN PREDICTION
# ============================================================

elif page == "Loan Prediction":

    st.header("🔮 Loan Approval Prediction")

    st.write(
        """
        Enter applicant information below to estimate
        the predicted loan status.
        """
    )

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    with col2:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=500,
            max_value=50000,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            max_value=50000,
            value=2000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=10,
            max_value=2000,
            value=200
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [120, 180, 240, 300, 360, 480]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1, 0]
        )

    # --------------------------------------------------------
    # TRAIN MODEL FOR PREDICTION
    # --------------------------------------------------------

    prediction_df = clean_df.copy()

    prediction_df = prediction_df.drop(
        columns=["Loan_ID"]
    )

    prediction_df["Loan_Status"] = (
        prediction_df["Loan_Status"]
        .map({
            "Approved": 1,
            "Rejected": 0
        })
    )

    X_pred_train = prediction_df.drop(
        columns=["Loan_Status"]
    )

    y_pred_train = prediction_df[
        "Loan_Status"
    ]

    for column in X_pred_train.select_dtypes(
        include=["object"]
    ).columns:

        X_pred_train[column] = (
            X_pred_train[column]
            .astype("category")
            .cat.codes
        )

    X_pred_train = X_pred_train.fillna(
        X_pred_train.median(
            numeric_only=True
        )
    )

    prediction_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    prediction_model.fit(
        X_pred_train,
        y_pred_train
    )

    # --------------------------------------------------------
    # CREATE USER DATA
    # --------------------------------------------------------

    user_data = pd.DataFrame({

        "Gender": [gender],

        "Married": [married],

        "Dependents": [dependents],

        "Education": [education],

        "Self_Employed": [self_employed],

        "ApplicantIncome": [
            applicant_income
        ],

        "CoapplicantIncome": [
            coapplicant_income
        ],

        "LoanAmount": [
            loan_amount
        ],

        "Loan_Amount_Term": [
            loan_term
        ],

        "Credit_History": [
            credit_history
        ],

        "Property_Area": [
            property_area
        ],

        "TotalIncome": [
            applicant_income
            + coapplicant_income
        ],

        "LoanIncomeRatio": [
            loan_amount /
            max(
                applicant_income
                + coapplicant_income,
                1
            )
        ]
    })

    for column in user_data.select_dtypes(
        include=["object"]
    ).columns:

        categories = (
            clean_df[column]
            .astype("category")
            .cat.categories
        )

        mapping = {
            category: index
            for index, category
            in enumerate(categories)
        }

        user_data[column] = (
            user_data[column]
            .map(mapping)
        )

    user_data = user_data.fillna(
        X_pred_train.median(
            numeric_only=True
        )
    )

    user_data = user_data[
        X_pred_train.columns
    ]

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "Predict Loan Approval"
    ):

        prediction = (
            prediction_model
            .predict(user_data)[0]
        )

        probability = (
            prediction_model
            .predict_proba(user_data)[0]
        )

        approval_probability = (
            probability[1] * 100
        )

        if prediction == 1:

            st.success(
                f"Loan Prediction: APPROVED"
            )

            st.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )

        else:

            st.error(
                f"Loan Prediction: REJECTED"
            )

            st.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header("💡 Business Insights")

    approval_rate = (
        (
            clean_df["Loan_Status"]
            == "Approved"
        ).mean()
        * 100
    )

    credit_approval = (
        clean_df
        .groupby("Credit_History")[
            "Loan_Status"
        ]
        .apply(
            lambda x:
            (x == "Approved").mean() * 100
        )
    )

    education_approval = (
        clean_df
        .groupby("Education")[
            "Loan_Status"
        ]
        .apply(
            lambda x:
            (x == "Approved").mean() * 100
        )
    )

    property_approval = (
        clean_df
        .groupby("Property_Area")[
            "Loan_Status"
        ]
        .apply(
            lambda x:
            (x == "Approved").mean() * 100
        )
    )

    st.subheader(
        "Key Project Findings"
    )

    st.write(
        f"""
        ### 1. Overall Approval Rate

        The overall loan approval rate in the analyzed
        dataset is approximately **{approval_rate:.2f}%**.
        """
    )

    st.write(
        """
        ### 2. Credit History

        Credit history is an important variable for analyzing
        loan approval patterns. Applicants with a positive
        credit history generally show a different approval
        pattern from applicants without a positive history.
        """
    )

    st.write(
        """
        ### 3. Applicant Income

        Applicant income can be compared with requested loan
        amount to understand the financial profile of applicants.
        """
    )

    st.write(
        """
        ### 4. Education

        Education level can be segmented to compare approval
        patterns between graduates and non-graduates.
        """
    )

    st.write(
        """
        ### 5. Property Area

        Urban, Semiurban and Rural applicants can be analyzed
        separately to identify differences in loan approval
        patterns.
        """
    )

    st.write(
        """
        ### 6. Data-Driven Decision Support

        The dashboard combines applicant information,
        financial information and credit-related variables
        to provide an analytical view of loan applications.
        """
    )

    st.subheader(
        "Approval Rate by Credit History"
    )

    st.dataframe(
        credit_approval.reset_index(
            name="Approval Rate (%)"
        )
    )

    st.subheader(
        "Approval Rate by Education"
    )

    st.dataframe(
        education_approval.reset_index(
            name="Approval Rate (%)"
        )
    )

    st.subheader(
        "Approval Rate by Property Area"
    )

    st.dataframe(
        property_approval.reset_index(
            name="Approval Rate (%)"
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Loan Approval Analytics | "
    "Python • SQL • Data Analytics • "
    "Machine Learning • Streamlit"
)
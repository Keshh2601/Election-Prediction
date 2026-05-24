import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Election Prediction System",
    page_icon="🗳️",
    layout="wide"
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("🗳️ Election Prediction System")
st.markdown("### Machine Learning Based Election Winner Prediction")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Option",
    [
        "Home",
        "Dataset",
        "Visualization",
        "Model Training",
        "Prediction",
        "About"
    ]
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Election Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ------------------------------------------------
    # CREATE WINNER COLUMN
    # ------------------------------------------------
    max_votes = df.groupby(
        ['year', 'pc_name']
    )['totvotpoll'].transform('max')

    df['winner'] = np.where(
        df['totvotpoll'] == max_votes,
        1,
        0
    )

    # ------------------------------------------------
    # REMOVE NULLS
    # ------------------------------------------------
    df.dropna(inplace=True)

    # ------------------------------------------------
    # ENCODE CATEGORICAL DATA
    # ------------------------------------------------
    encoder = LabelEncoder()

    categorical_columns = [
        'st_name',
        'pc_name',
        'pc_type',
        'cand_name',
        'cand_sex',
        'partyname',
        'partyabbre'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])

    # ------------------------------------------------
    # FEATURES
    # ------------------------------------------------
    X = df[
        [
            'st_name',
            'year',
            'pc_no',
            'pc_name',
            'pc_type',
            'cand_sex',
            'partyname',
            'partyabbre',
            'totvotpoll',
            'electors'
        ]
    ]

    y = df['winner']

    # ------------------------------------------------
    # SPLIT DATA
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ------------------------------------------------
    # HOME PAGE
    # ------------------------------------------------
    if menu == "Home":

        st.header("🏠 Home")

        st.write("""
        This application predicts election winners
        using Machine Learning algorithms.
        """)

        st.subheader("Algorithms Used")

        st.write("""
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - XGBoost
        """)

    # ------------------------------------------------
    # DATASET PAGE
    # ------------------------------------------------
    elif menu == "Dataset":

        st.header("📊 Dataset")

        st.write("Dataset Shape:")
        st.write(df.shape)

        st.write("Columns:")
        st.write(df.columns)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    # ------------------------------------------------
    # VISUALIZATION PAGE
    # ------------------------------------------------
    elif menu == "Visualization":

        st.header("📈 Data Visualization")

        chart_option = st.selectbox(
            "Select Visualization",
            [
                "Votes Distribution",
                "Winner Distribution",
                "Gender Distribution"
            ]
        )

        # Votes Distribution
        if chart_option == "Votes Distribution":

            fig, ax = plt.subplots()

            sns.histplot(
                df['totvotpoll'],
                bins=30,
                ax=ax
            )

            ax.set_title("Votes Distribution")

            st.pyplot(fig)

        # Winner Distribution
        elif chart_option == "Winner Distribution":

            fig, ax = plt.subplots()

            sns.countplot(
                x='winner',
                data=df,
                ax=ax
            )

            ax.set_title("Winner vs Loser")

            st.pyplot(fig)

        # Gender Distribution
        elif chart_option == "Gender Distribution":

            fig, ax = plt.subplots()

            sns.countplot(
                x='cand_sex',
                data=df,
                ax=ax
            )

            ax.set_title("Candidate Gender Distribution")

            st.pyplot(fig)

    # ------------------------------------------------
    # MODEL TRAINING PAGE
    # ------------------------------------------------
    elif menu == "Model Training":

        st.header("🤖 Model Training")

        model_option = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost"
            ]
        )

        # Logistic Regression
        if model_option == "Logistic Regression":

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(max_iter=5000)

            model.fit(X_train_scaled, y_train)

            predictions = model.predict(X_test_scaled)

        # Decision Tree
        elif model_option == "Decision Tree":

            model = DecisionTreeClassifier()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

        # Random Forest
        elif model_option == "Random Forest":

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

        # XGBoost
        elif model_option == "XGBoost":

            model = XGBClassifier()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.success(f"Accuracy: {accuracy:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            ax=ax
        )

        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions
        )

        st.text(report)

    # ------------------------------------------------
    # PREDICTION PAGE
    # ------------------------------------------------
    elif menu == "Prediction":

        st.header("🧠 Election Prediction")

        st.subheader("Enter Candidate Details")

        st_name = st.number_input(
            "State Code",
            min_value=0
        )

        year = st.number_input(
            "Election Year",
            min_value=2000
        )

        pc_no = st.number_input(
            "Constituency Number",
            min_value=0
        )

        pc_name = st.number_input(
            "Constituency Code",
            min_value=0
        )

        pc_type = st.number_input(
            "PC Type",
            min_value=0
        )

        cand_sex = st.number_input(
            "Candidate Gender Code",
            min_value=0
        )

        partyname = st.number_input(
            "Party Code",
            min_value=0
        )

        partyabbre = st.number_input(
            "Party Abbreviation Code",
            min_value=0
        )

        totvotpoll = st.number_input(
            "Total Votes Polled",
            min_value=0
        )

        electors = st.number_input(
            "Number of Electors",
            min_value=0
        )

        predict_model = st.selectbox(
            "Select Prediction Model",
            [
                "Random Forest",
                "Decision Tree",
                "Logistic Regression",
                "XGBoost"
            ]
        )

        if st.button("Predict Result"):

            sample_input = pd.DataFrame(
                [[
                    st_name,
                    year,
                    pc_no,
                    pc_name,
                    pc_type,
                    cand_sex,
                    partyname,
                    partyabbre,
                    totvotpoll,
                    electors
                ]],
                columns=[
                    'st_name',
                    'year',
                    'pc_no',
                    'pc_name',
                    'pc_type',
                    'cand_sex',
                    'partyname',
                    'partyabbre',
                    'totvotpoll',
                    'electors'
                ]
            )

            # Train Selected Model
            if predict_model == "Random Forest":

                model = RandomForestClassifier()

                model.fit(X_train, y_train)

                prediction = model.predict(sample_input)

            elif predict_model == "Decision Tree":

                model = DecisionTreeClassifier()

                model.fit(X_train, y_train)

                prediction = model.predict(sample_input)

            elif predict_model == "Logistic Regression":

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(X_train)

                sample_scaled = scaler.transform(sample_input)

                model = LogisticRegression(max_iter=5000)

                model.fit(X_train_scaled, y_train)

                prediction = model.predict(sample_scaled)

            else:

                model = XGBClassifier()

                model.fit(X_train, y_train)

                prediction = model.predict(sample_input)

            if prediction[0] == 1:
                st.success("🏆 Predicted Result: WINNER")
            else:
                st.error("❌ Predicted Result: LOSER")

    # ------------------------------------------------
    # ABOUT PAGE
    # ------------------------------------------------
    elif menu == "About":

        st.header("ℹ️ About Project")

        st.write("""
        Election Prediction System developed using:

        - Python
        - Streamlit
        - Machine Learning
        - Scikit-Learn
        - XGBoost
        - Pandas
        - Matplotlib
        - Seaborn
        """)

else:

    st.warning("Please upload a CSV dataset.")

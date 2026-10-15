
import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Fake Job Posting Detection System")
st.write(
    "Analyze a job advertisement using Machine Learning "
    "to estimate whether it may be fake or genuine."
)

PROJECT_DIR = (
    r"C:\Users\Biswajit Mandal"
    r"\AI-Powered Fake Job Posting Detection and risk Assessment System"
)

DATA_PATH = os.path.join(
    PROJECT_DIR, " Data", "fake_job_postings.csv"
)

@st.cache_resource
def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "fraudulent" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'fraudulent' columns."
        )

    df = df[["text", "fraudulent"]].dropna()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["fraudulent"],
        test_size=0.33,
        random_state=53,
        stratify=df["fraudulent"]
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    return vectorizer, model, accuracy, len(df)

try:
    with st.spinner("Training the model on your dataset..."):
        vectorizer, model, accuracy, rows = train_model()

    col1, col2 = st.columns(2)
    col1.metric("Training Dataset Records", f"{rows:,}")
    col2.metric("Test Accuracy", f"{accuracy:.2%}")

    st.success("Machine Learning model is ready.")

    st.subheader("Enter Job Advertisement Details")

    title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    location = st.text_input("Job Location")
    description = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the job description here..."
    )

    requirements = st.text_area(
        "Requirements (optional)",
        height=100
    )

    benefits = st.text_area(
        "Benefits (optional)",
        height=80
    )

    if st.button("Analyze Job Posting", type="primary"):
        if not description.strip():
            st.warning("Please enter a job description.")
        else:
            job_text = " ".join([
                title, company, location,
                description, requirements, benefits
            ]).strip()

            job_vector = vectorizer.transform([job_text])
            prediction = int(model.predict(job_vector)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(job_vector)[0]
                classes = list(model.classes_)
                fake_probability = float(
                    probabilities[classes.index(1)]
                ) if 1 in classes else 0.0
            else:
                fake_probability = None

            st.subheader("Analysis Result")

            if prediction == 1:
                st.error("⚠️ The model classifies this posting as potentially FAKE.")
            else:
                st.success("✅ The model classifies this posting as likely GENUINE.")

            if fake_probability is not None:
                st.metric(
                    "Model-estimated Fake Probability",
                    f"{fake_probability:.2%}"
                )

            st.caption(
                "This is a machine-learning prediction, not a guarantee. "
                "Verify the employer, official website, contact details, "
                "and never pay recruitment fees based only on this result."
            )

except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.info(
        "Check that the dataset is inside the project's ' Data' folder "
        "and contains the 'text' and 'fraudulent' columns."
    )

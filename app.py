from pathlib import Path

import joblib
import streamlit as st

folder = Path(__file__).parent

model = joblib.load(folder / "physics_model.pkl")
vectorizer = joblib.load(folder / "tfidf_vectorizer.pkl")

st.title("Physics Paper Classifier")

abstract = st.text_area("Paste a paper abstract:")

if st.button("Predict"):
    if abstract.strip():
        scores = vectorizer.transform([abstract])

        if scores.nnz == 0:
            st.warning("No recognised words. Try another abstract.")
        else:
            prediction = model.predict(scores)[0]
            st.success(f"Category: {prediction}")
    else:
        st.warning("Enter an abstract first.")
import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("physics_classifier.joblib")

st.title("Physics Paper Classifier")

st.write(
    "Classify physics research papers using NLP "
    "and machine learning."
)

# Display supported categories
categories = model.named_steps["classifier"].classes_

st.write("Supported categories:")
st.write(", ".join(categories))

# User enters abstract
abstract = st.text_area(
    "Enter a physics paper abstract",
    height=200
)

if st.button("Classify"):

    if abstract.strip():

        # Predict category
        prediction = model.predict([abstract])[0]

        # Predict probabilities
        probabilities = model.predict_proba([abstract])[0]

        st.subheader("Prediction")
        st.success(prediction)

        # Display probabilities
        results = pd.DataFrame({
            "Category": categories,
            "Probability": probabilities
        })

        st.subheader("Prediction Probabilities")

        st.dataframe(results, hide_index=True)

        st.bar_chart(
            results.set_index("Category")
        )

    else:
        st.warning("Please enter an abstract.")

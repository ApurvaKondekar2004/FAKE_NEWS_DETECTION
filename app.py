
import streamlit as st
from utils import predict_news

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection (Domain Transfer Models)")

headline = st.text_input("Enter a news headline:")

model_choice = st.selectbox(
    "Choose a model:",
    ("model_POL (trained on PolitiFact)", "model_GOSSIP (trained on GossipCop)")
)

if st.button("Predict"):
    if headline.strip():
        model_path = "model_POL" if "POL" in model_choice else "model_GOSSIP"
        result = predict_news(headline, model_path)

        st.subheader("Prediction")
        st.write(f"**Label:** {result['label']}")
        st.write(f"**Confidence:** {result['confidence'] * 100:.2f}%")

        # Show probability bar
        st.progress(result['confidence'])
    else:
        st.warning("Please enter a headline before predicting.")

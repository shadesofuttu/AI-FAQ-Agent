import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load data
faq = pd.read_csv("faq.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

faq_questions = faq["question"].tolist()
faq_embeddings = model.encode(faq_questions)

# UI
st.title("🤖 UTTU's AI FAQ Agent")

user_question = st.text_input("Ask a question")

if user_question:

    user_embedding = model.encode(user_question)

    scores = cos_sim(user_embedding, faq_embeddings)[0]

    best_match_index = scores.argmax().item()

answer = faq.iloc[best_match_index]["answer"]

confidence = scores[best_match_index].item()

if confidence < 0.5:
    st.error("Sorry, I couldn't find a relevant answer.")
else:
    st.success(answer)
    st.write(f"Confidence Score: {confidence:.2f}")
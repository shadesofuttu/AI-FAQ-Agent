import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load data
model = SentenceTransformer("all-MiniLM-L6-v2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


uploaded_file = st.file_uploader(
    "Upload your FAQ CSV",
    type=["csv"]
)

if uploaded_file is not None:

    faq = pd.read_csv(uploaded_file)

    st.success("CSV Loaded Successfully!")

    faq_questions = faq["question"].tolist()

    faq_embeddings = model.encode(faq_questions)

    st.title("🤖 UTTU's AI FAQ Agent")

    user_question = st.text_input("Ask a question")

    if user_question:

        user_embedding = model.encode(user_question)

        scores = cos_sim(user_embedding, faq_embeddings)[0]

        best_match_index = scores.argmax().item()

        answer = faq.iloc[best_match_index]["answer"]

        st.session_state.chat_history.append(
     {
        "question": user_question,
        "answer": answer
     }
)

        confidence = scores[best_match_index].item()

        if confidence < 0.5:
            st.error("Sorry, I couldn't find a relevant answer.")
        else:
            st.success(answer)
            st.write(f"Confidence Score: {confidence:.2f}")

st.subheader("Chat History")

for chat in st.session_state.chat_history:

    st.write(f"🧑 You: {chat['question']}")
    st.write(f"🤖 Bot: {chat['answer']}")
       
else:
    st.info("Please upload a FAQ CSV file to begin.")
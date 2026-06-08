import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# SIDEBAR
st.sidebar.title("🤖 AI FAQ Agent")

st.sidebar.write(
    "Upload a FAQ CSV and ask questions."
)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# CACHING
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


@st.cache_data
def create_embeddings(questions):
    return model.encode(questions)
# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload your FAQ CSV",
    type=["csv"]
)


if uploaded_file is not None:

    faq = pd.read_csv(uploaded_file)

    st.success("CSV Loaded Successfully!")

    faq_questions = faq["question"].tolist()

    faq_embeddings = create_embeddings(faq_questions)
# UI
    st.title("🤖 UTTU's AI FAQ Agent")

    st.caption(
        "Semantic Search powered by Sentence Transformers"
    )

    user_question = st.text_input("Ask a question")

    if user_question:

        # User embedding
        user_embedding = model.encode(user_question)

        # Similarity scores
        scores = cos_sim(
            user_embedding,
            faq_embeddings
        )[0]

        best_match_index = scores.argmax().item()

        confidence = scores[
            best_match_index
        ].item()

        if confidence < 0.5:

            st.error(
                "Sorry, I couldn't find a relevant answer."
            )

        else:

            answer = faq.iloc[
                best_match_index
            ]["answer"]

            st.success(answer)

            st.write(
                f"Confidence Score: {confidence:.2f}"
            )

            # Save only valid answers
            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                }
            )

else:

    st.info(
        "Please upload a FAQ CSV file to begin."
    )
if st.session_state.chat_history:

    st.subheader("Chat History")

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])
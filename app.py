import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import google.generativeai as genai
# GEMINI CONFIG

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)
gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)
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
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
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

    st.success(
        "CSV Loaded Successfully!"
    )

    faq_questions = faq[
        "question"
    ].tolist()

    faq_embeddings = create_embeddings(
        faq_questions
    )

    # UI

    st.title(
        "🤖 UTTU's AI FAQ Agent"
    )

    st.caption(
        "Semantic Search + Gemini AI"
    )

    user_question = st.text_input(
        "Ask a question"
    )

    if user_question:

        # User embedding
        user_embedding = model.encode(
            user_question
        )

        # Similarity scores
        scores = cos_sim(
            user_embedding,
            faq_embeddings
        )[0]

        best_match_index = (
            scores.argmax().item()
        )

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

            # GEMINI REWRITE

            gemini_response = (
                gemini_model.generate_content(
                    f"""
                    Rewrite this FAQ answer in a friendly and professional way.

                    FAQ Answer:
                    {answer}

                    Keep the meaning exactly the same.
                    Do not add any new information.
                    """
                )
            )
            try:

                gemini_response = gemini_model.generate_content(
                f"""
                Rewrite the following FAQ answer in a friendly and professional tone.

                FAQ Answer:
                {answer}

                Rules:
                - Return ONLY one rewritten answer.
                - Do not provide multiple options.
                - Do not use bullet points.
                - Do not add new information.
                - Keep the meaning exactly the same.
                """
)
            

                final_answer = gemini_response.text

            except Exception:

                final_answer = answer
            st.success(
                final_answer
            )

            st.write(
                f"Confidence Score: {confidence:.2f}"
            )

            # Save answer to history

            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": final_answer,
                }
            )

else:

    st.info(
        "Please upload a FAQ CSV file to begin."
    )


# CHAT HISTORY

if st.session_state.chat_history:

    st.subheader(
        "Chat History"
    )

    for chat in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            "user"
        ):
            st.write(
                chat["question"]
            )

        with st.chat_message(
            "assistant"
        ):
            st.write(
                chat["answer"]
            )
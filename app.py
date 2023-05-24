import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from lang_utils import ask_to_all_pdfs_sources, create_qa_retrievals

# SETUP ------------------------------------------------------------------------
favicon = Image.open("favicon.ico")
st.set_page_config(
    page_title="PDF Comparison - LLM",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="auto",
)


# Sidebar contents ------------------------------------------------------------------------
with st.sidebar:
    st.title("LLM - PDF Comparison App")
    st.markdown(
        """
    ## About
    This app is an pdf comparison (LLM-powered), built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI LLM model](https://platform.openai.com/docs/models) 
    """
    )
    st.write(
        "Made with ❤️ by [Chasquilla Engineer](https://resume.chasquillaengineer.com/)"
    )


# ROW 1 ------------------------------------------------------------------------

Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 300vw 300vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1>Super PDF Comparison</h1>
    </div>
    """
components.html(Title_html)

with st.form("basic_form"):

    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf"],
        key="file_upload_widget",
        accept_multiple_files=True,
    )

    question_1 = st.text_input("Question 1", key="1_question")
    question_2 = st.text_input("Question 2", key="2_question")
    question_3 = st.text_input("Question 3", key="3_question")
    # question_4 = st.text_input("Question 4", key="4_question")
    # question_5 = st.text_input("Question 5", key="5_question")

    submit_btn = st.form_submit_button("Start Processing")

    if submit_btn:
        if openai_api_key == "":
            st.warning("You need an API key from OpenAI to use thise App")
            st.stop()

        if question_1 == "":
            st.warning("Give at least one question")
            st.stop()

        if uploaded_files is None:
            st.warning("Upload at least 1 PDf file")
            st.stop()
        all_questions = [question_1, question_2, question_3]  # question_4, question_5]
        with st.spinner("Creating embeddings...."):
            try:

                st.session_state.qa_retrievals = create_qa_retrievals(
                    uploaded_files, openai_api_key
                )
                st.session_state.questions = all_questions
            except Exception as e:

                st.error("Something went grong...")
                st.exception(e)
                st.stop()
        st.success("Done!", icon="✅")
        with st.spinner("Doing Analysis...."):

            try:
                data = []
                for question in st.session_state.questions:
                    if question == "":
                        continue
                    results = ask_to_all_pdfs_sources(
                        question, st.session_state.qa_retrievals
                    )
                    data.extend(results)
                st.write(data)
                st.session_state.data = data

            except Exception as e:

                st.error("Something went grong...")
                st.exception(e)
                st.stop()
        st.success("Done!", icon="✅")
        with st.spinner("Doing Analysis.."):
            try:
                df = pd.DataFrame(st.session_state.data)
                st.table(
                    df.pivot(
                        index="query", columns="source_document", values="response"
                    )
                )

            except Exception as e:

                st.error("Something went grong...")
                st.exception(e)
                st.stop()
        st.snow()

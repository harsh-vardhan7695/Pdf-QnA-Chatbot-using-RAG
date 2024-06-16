from rag import RAG
import streamlit as st
import asyncio

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'vectorstore_created' not in st.session_state:
    st.session_state.vectorstore_created = False
# flag = False

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    
loop = get_or_create_eventloop()
#asyncio.set_event_loop(loop)

@st.cache_resource(show_spinner = True)
def load_rag_pipeline(uploaded_files ):
    rag_pipeline=  RAG(uploaded_files )
    return rag_pipeline

st.title("RAG Project")
st.subheader("This is end to end RAG pipeline")

with st.sidebar:
        st.title("Menu:")
        uploaded_files  = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                rag_pipe = load_rag_pipeline(uploaded_files )
                st.session_state.clicked = True
                st.session_state.rag_pipe = rag_pipe

if st.session_state.clicked:
    st.subheader("Enter the question")
    question = st.text_input("Question")
    if question:
        answer, vs = st.session_state.rag_pipe.qa(question, st.session_state.vectorstore_created)
        st.session_state.vectorstore_created = vs
        st.subheader("Answer")
        # answer = rag_pipe.get_answer(question)
        st.write(answer)
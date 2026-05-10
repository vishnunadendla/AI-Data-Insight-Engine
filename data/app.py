import streamlit as st

st.title("Chat with Vishnu's AI Agent")

# Get question from user
question = st.text_input("Ask me anything about Vishnu:")

# When question is entered
if question:
    answer = answer_question(???, index, Resume_Chunks)
    st.write(answer)
# from transformers import pipeline
# generator = pipeline('text-generation', model='gpt2')
# user_prompt = "The quick brown fox jumps over the lazy"
# response = generator(user_prompt, max_new_tokens=50, num_return_sequences=1, clean_up_tokenization_spaces=True)
# print("Your Prompt:", user_prompt)
# print("LLM's Response:", response[0]['generated_text'])


import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from agent import get_agent_executor
from image import imgg

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="AI Research Agent 📚", page_icon="🤖", layout="wide")

# Custom CSS for chat messages
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #E6F3FF;
    }
    .stChatMessage.assistant {
        background-color: #F0F0F0;
    }
    </style>
""", unsafe_allow_html=True)
st.title("📚 Text Generative AI")

# st.title("📚 Text Generative AI")
st.caption("Feel free to ask!!. Your advanced AI assistant to provide answers based on internal knowledge.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message_obj in st.session_state.chat_history:
    role = "user" if isinstance(message_obj, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message_obj.content)

# User input
user_query = st.chat_input("Ask a research question...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                agent_executor = get_agent_executor(user_query)
                print('agent excutpore answer-->',agent_executor)
                # response = agent_executor.invoke({
                #     "input": user_query,
                #     "chat_history": st.session_state.chat_history[:-1]
                # })
                answer=agent_executor
                # answer = response["output"]
                print('response',answer)
                # st.image(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.markdown(answer)

            except Exception as e:
                error_message = f"😕 Apologies, an error occurred: {str(e)}"
                st.error(error_message)
                print(f"Error during agent invocation: {e}")
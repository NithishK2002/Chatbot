import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import time
import os
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
  ForgotError,
  LoginError,
  RegisterError,
  ResetError,
  UpdateError)
# from chatbot import chatBot


# import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# st.markdown(
#     """
#   <style>
#       section[data-testid="stSidebar"] {
#           width: 10px !important; # Set the width to your desired value
#       }
#       .stApp{
#         width: 1000px;
#         height: 800px;
#       }
#   </style>
#   """,unsafe_allow_html=True,
# )
st.set_page_config(page_title="Government Infobot", page_icon="🤖")
st.title("Government Infobot")

def get_vectorstore_from_url():
  # output_directory=os.getcwd()
  # print(output_directory)
  # output_directory_pdf=output_directory+"/chroma_db_pdf"
  # files=os.listdir(output_directory_pdf)
  # # print(files)
  # db_file=files[1]
  # print(db_file)
  # # def get_vectorstore_from_disk(output_directory, vector_store_index):
  # vector_store_file = os.path.join(output_directory, db_file)
  # vector_store = Chroma(persist_directory=vector_store_file,embedding_function=OpenAIEmbeddings())  # Corrected line
  # output_directory_url=output_directory+"/chroma_db"
  # files=os.listdir(output_directory_url)
  # db_file=files[1]
  # print(db_file)
  # vector_store_file = os.path.join(output_directory, db_file)
  # vector_store = Chroma(persist_directory=vector_store_file,embedding_function=OpenAIEmbeddings())
  pdf_vector_store = FAISS.load_local("./faiss_pdf_1",
                                      OpenAIEmbeddings(),
                                      allow_dangerous_deserialization=True)
  url_vector_store = FAISS.load_local("./faiss_url_1",
                                      OpenAIEmbeddings(),
                                      allow_dangerous_deserialization=True)
  pdf_vector_store.merge_from(url_vector_store)
  vector_store = pdf_vector_store
  # vector_store=vector_store.append(url_vector_store)
  # url_vector_store=url_vector_store.append(pdf_vector_store)
  # vector_store= combine_vector_stores(url_vector_store,pdf_vector_store)
  return vector_store


def get_context_retriever_chain(vector_store):
  llm = ChatOpenAI()

  retriever = vector_store.as_retriever()

  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
      ("user",
       "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"
       )
  ])

  retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

  return retriever_chain


def get_conversational_rag_chain(retriever_chain):

  llm = ChatOpenAI()

  prompt = ChatPromptTemplate.from_messages([
      ("system",
       "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
  ])

  stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

  return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
  retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
  conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

  response = conversation_rag_chain.invoke({
      "chat_history": st.session_state.chat_history,
      "input": user_input
  })

  return response['answer']


# app config
def logout_button():
  st.query_params.key = "login"
  st.session_state['logout'] = True
  st.session_state['name'] = None
  st.session_state['username'] = None
  st.session_state['authentication_status'] = None
  st.session_state['app_page'] = None


def chatBot():
  try:
    st.session_state['app_page'] = True
    username = st.session_state['name']
    # username = "nithish"
    username = username.capitalize()
    with st.sidebar:
      st.title(f"Hello, {username}")
      st.button("Logout", on_click=logout_button)
    # print(st.session_state['authentication_status'])
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
          AIMessage(
              content=
              f"Hello, I am a Government Infobot. How can I help you {username}?"
          ),
      ]
    if "vector_store" not in st.session_state:
      st.session_state.vector_store = get_vectorstore_from_url()
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
      response = get_response(user_query)
      st.session_state.chat_history.append(HumanMessage(content=user_query))
      st.session_state.chat_history.append(AIMessage(content=response))

      # conversation
    for message in st.session_state.chat_history:
      if isinstance(message, AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
          st.write(message.content)
  except Exception as e:
    st.rerun()


# Check if set_page_config has been executed already
if not st.session_state.get("has_set_page_config", False):
    # Execute set_page_config
    
    # Set the flag to indicate that set_page_config has been executed
    st.session_state["has_set_page_config"] = True
# st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
#                       'Get Help': 'https://www.extremelycoolapp.com/help',
#                       'Report a bug': "https://www.extremelycoolapp.com/bug",
#                       'About': "# This is a header. This is an *extremely* cool app!"
#                   })
# st.title("Government Infobot")


def saveToYaml():
    with open('./config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)
# Loading config file
with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)


# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# st.markdown(
#     """
#     <style>
#         section[data-testid="stSidebar"] {
#             width: 80px !important; # Set the width to your desired value
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
st.markdown(
    """
  <title>Chat with websites</title>
  <style>
      html {
        font-size: 20px;
      }
      section[data-testid="stSidebar"] {
          width: 10px !important; # Set the width to your desired value
      }
      /*.st-emotion-cache-1eo1tir {
        padding: 1rem 1rem 1rem 0rem;
      }
      .st-emotion-cache-arzcut {
        padding: 1rem 1rem 1rem 0rem;
      }*/
  </style>
  """,unsafe_allow_html=True,
)
def logout_button():
    st.query_params.key="login"
    st.session_state['logout'] = True
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.session_state['authentication_status'] = None
    # st.rerun()

def app():
    # st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
    #                       'Get Help': 'https://www.extremelycoolapp.com/help',
    #                       'Report a bug': "https://www.extremelycoolapp.com/bug",
    #                       'About': "# This is a header. This is an *extremely* cool app!"
    #                   })
    # st.title("Government Infobot")
    # st.write("Welcome to the Government Infobot")
    
    chatBot()

def login():
    try:
        # print(st.session_state['authentication_status'])
        (name_of_the_user,login_status,username)=authenticator.login()
        # print("Login Status",login_status)
        if st.session_state['authentication_status']:
            st.success("Logged in successfully!")
            # time.sleep(5)
            st.query_params.key="app"

            # st.rerun()
    except LoginError as e:
        st.error(e)
    st.markdown("Don't have an account? [Register here](/?key=register)")

def register():
    try:
        (email_of_registered_user,username_of_registered_user,name_of_registered_user) = authenticator.register_user(pre_authorization=False)
        if email_of_registered_user:
            st.success('User registered successfully')
            saveToYaml()
            st.query_params.key="login"
            st.rerun()

    except RegisterError as e:
        st.error(e)
    st.markdown("Done Registering? [Login here](/?key=login)")


def main():
    if "key" not in st.query_params or st.query_params.key == "login":
        st.query_params.key = "login"
        st.session_state['logout'] = True
        st.session_state['name'] = None
        st.session_state['username'] = None
        st.session_state['authentication_status'] = None
        st.session_state['app_page'] = None
    page = st.query_params["key"]
    print(page)
    if page == "login" and st.session_state['authentication_status'] == None and st.session_state['app_page'] == None:
        login()
    elif page == "register":
        register()
    elif page == "app":
        app()


if __name__ == "__main__":
    # st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
    #                                         'Get Help': 'https://www.extremelycoolapp.com/help',
    #                                         'Report a bug': "https://www.extremelycoolapp.com/bug",
    #                                         'About': "# This is a header. This is an *extremely* cool app!"
    #                                     })
    # st.title("Government Infobot")
    main()

# TODO #1: Import necessary libraries
import os
import pandas as pd
import taipy.gui.builder as tgb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from taipy.gui import Gui, notify

# TODO #2: Load environment variables
load_dotenv()

# TODO #3: Define the folder path for the FAQ text file
data_dir = './data'

# TODO #4: Initialize the data loader
loader = DirectoryLoader(data_dir, glob='*.txt')
docs = loader.load()

# TODO #5: Create the document index
index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
).from_documents(docs)

# TODO #6: Configure the LLM and retrieval chain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_tokens=512,
    timeout=None,
    max_retries=2,
)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question",
)

# TODO #7: Define the function to query the LLM
def query_llm(query_message):
    return chain.run(query_message)

# TODO #8: Initialize chatbot state variables
query_message = ""
messages = []
messages_dict = {}


def on_init(state):
    state.messages = [
        {
            "style": "assistant_message",
            "content": "Hi, I am StackUp assistant! How can I help you today?",
        },
    ]
    new_conv = create_conv(state)
    state.conv.update_content(state, new_conv)

# TODO #9: Define helper functions for the chatbot
def create_conv(state):
    messages_dict = {}
    with tgb.Page() as conversation:
        for i, message in enumerate(state.messages):
            text = message["content"].replace("<br>", "\n").replace('"', "'")
            messages_dict[f"message_{i}"] = text
            tgb.text(
                f"{{messages_dict.get('message_{i}') or ''}}",
                class_name=f"message_base {message['style']}",
                mode="md",
            )
    state.messages_dict = messages_dict
    return conversation


def send_message(state):
    state.messages.append(
        {
            "style": "user_message",
            "content": state.query_message,
        }
    )
    state.conv.update_content(state, create_conv(state))
    notify(state, "info", "Sending message...")
    state.messages.append(
        {
            "style": "assistant_message",
            "content": query_llm(state.query_message),
        }
    )
    state.conv.update_content(state, create_conv(state))
    state.query_message = ""


def reset_chat(state):
    state.query_message = ""
    on_init(state)

# TODO #10: Design the GUI layout
with tgb.Page() as page:
    with tgb.layout(columns="350px 1"):
        with tgb.part(class_name="sidebar"):
            tgb.text("## StackUp Assistant", mode="md")
            tgb.button(
                "New Conversation",
                class_name="fullwidth plain",
                on_action=reset_chat,
            )

        with tgb.part(class_name="p1"):
            tgb.part(partial="{conv}", height="600px", class_name="card card_chat")
            with tgb.part("card mt1"):
                tgb.input(
                    "{query_message}",
                    on_action=send_message,
                    change_delay=-1,
                    label="Write your message:",
                    class_name="fullwidth",
                )

# TODO #11: Add the application run logic
if __name__ == "__main__":
    gui = Gui(page)
    conv = gui.add_partial("")
    gui.run(title="StackUp Assistant", dark_mode=False, margin="0px", debug=True)

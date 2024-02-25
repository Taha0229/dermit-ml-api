from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import json
import os
import openai
from dotenv import load_dotenv
# from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
# from langchain.prompts import HumanMessagePromptTemplate
# from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware  

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins instead of "*" for all
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
load_dotenv()


openai.api_key = os.getenv('OPENAI_API_KEY') 



documents = SimpleDirectoryReader("./data").load_data()

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("dermit")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

llm_model = "gpt-3.5-turbo-0301"
Settings.llm = OpenAI(temperature=0.0, model=llm_model, context_window=3900, max_new_tokens=1500)



response_schemas = [
    ResponseSchema(
        name="Name",
        description=(
            "Name of identitfied disease."
        ),
    ),
    ResponseSchema(
        name="Symptoms",
        description="Explain each symptom in pointwise manner.",
    ),
    ResponseSchema(
        name="Cause",
        description="Elaborate the causes of the disease.",
    ),
    ResponseSchema(
        name="Lifestyle changes",
        description="Describes the possible life stye changes that can be subscribed to prevent the disease.",
    ),
    ResponseSchema(
        name="Background of the disease:",
        description="Describe the background of the diagnosed disease.",
    )
]

lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)



# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.USER,
        content=(
            '''You are a professional dermatologist. You will be provided with \
            patient's input query based on which your goal is to generate \
            response as accurately as possible based on the context provided.\
            If you don't know the answer, just say that you don't know.'''
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "diagnose the disease: {query_str}\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs, output_parser=output_parser)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.USER,
        content=(
            '''You are a professional dermatologist. You will be provided with \
            patient's input query based on which your goal is to generate \
            response as accurately as possible based on the context provided.\
            If you don't know the answer, just say that you don't know.'''
            "We have the opportunity to refine the original answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question: {query_str}. "
            "If the context isn't useful, output the original answer again.\n"
            "Original Answer: {existing_answer}"
        ),
    ),
]
refine_template = ChatPromptTemplate(chat_refine_msgs, output_parser=output_parser)

query_engine = index.as_query_engine(text_qa_template=text_qa_template, refine_template=refine_template)



class QueryInput(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hello": "World, this is root route"}

@app.post("/process_query")
async def process_query(query_input: QueryInput):
    # Get the user query from the request body
    user_query = query_input.query
    response = query_engine.query(user_query)
    op = response.response
    op = op.replace("```", "").replace("json", "")
    op_json = json.loads(op)
    
    
    return op_json


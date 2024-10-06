### Import Section ###
from operator import itemgetter
import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough


import chainlit as cl


### Global Section ###
from dotenv import load_dotenv; _ = load_dotenv()

# Ideally this is all pre-indexed as in the midterm here: https://github.com/dhrits/ai-ethics-bot/blob/main/app.py#L44
# But just for this week's "homework", it is ok to do this because it's a single document
# which needs indexing
FILE_PATH = "./eu_ai_act.pdf"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
Loader = PyMuPDFLoader
loader = Loader(FILE_PATH)
documents = loader.load()
docs = text_splitter.split_documents(documents)
for i, doc in enumerate(docs):
    doc.metadata["source"] = f"source_{i}"
# Typical Embedding Model
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Typical QDrant Client Set-up
collection_name = f"{FILE_PATH}_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Adding cache!
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings, store, namespace=core_embeddings.model
)

# Typical QDrant Vector Store Set-up
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=cached_embedder)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


def get_rag_chain():
    global retriever
    """Gets a RAG chain"""

    rag_system_prompt_template = """\
    You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
    """

    rag_message_list = [
        {"role" : "system", "content" : rag_system_prompt_template},
    ]

    rag_user_prompt_template = """\
    Question:
    {question}
    Context:
    {context}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt_template),
        ("human", rag_user_prompt_template)
    ])

    chat_model = ChatOpenAI(model="gpt-4o-mini")
    set_llm_cache(InMemoryCache())

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | chat_model
    )
    return retrieval_augmented_qa_chain

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    """Initialization of the application"""
    msg = cl.Message(
        content="", disable_human_feedback=True
    )
    await msg.send()
    chain = get_rag_chain()
    # Let the user know that the system is ready
    msg.content = """
    I'm ready to answer any of your questions regarding the EU AI Act.
    Ask away!
    """
    await msg.update()

    cl.user_session.set("chain", chain)

# Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ Renames the chatbot """
    rename_dict = {
        'Chatbot': 'EU-AI-Act-Bot',
        'User': 'User',
    }
    return rename_dict.get(orig_author, orig_author)

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """Run on user message"""
    chain = cl.user_session.get("chain")
    
    msg = cl.Message(content="")
    res = chain.astream({"question": message.content})
    async for resp in res:
        await msg.stream_token(resp.content)

    await msg.send()

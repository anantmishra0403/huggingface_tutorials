from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name="BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name, token=hf_token)

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="alfred_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

reader = SimpleDirectoryReader(input_dir='/Users/anantmishra/projects/huggingface_tutorials/resources')
documents = reader.load_data()

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap = 0),
        embed_model
    ],
    vector_store=vector_store
)

pipeline.run(documents = documents)


index = VectorStoreIndex.from_vector_store(vector_store,embed_model = embed_model)

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)

query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")
result = query_engine.query("what is attention in deep learning models?")
print(result)



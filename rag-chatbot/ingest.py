from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
loader = TextLoader("data/docs.txt", encoding="utf-8")
documents = loader.load()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embeddings model (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)

# Save DB locally
db.save_local("vectorstore/faiss_index")

print("Vector DB created successfully ✔")
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


#chunk_size 가 크면 retriever가 더 긴 문서를 반환하지만, chunk_size가 작으면 retriever가 더 짧은 문서를 반환한다. 
# chunk_overlap이 크면 chunk 간의 중복이 많아져서 retriever가 더 긴 문서를 반환할 가능성이 높아진다.
#  따라서 chunk_size와 chunk_overlap을 적절히 조절하여 retriever가 원하는 길이의 문서를 반환하도록 할 수 있다.

def create_chunks(documents):
    #Splitter 는 다양한 종류가 있음
    # Split Markdown , JSON , code , HTML
    # https://docs.langchain.com/oss/python/integrations/splitters

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


# 실행하면 vector_db 폴더에 임베딩이 저장된다. 이후에는 이 폴더를 불러와서 retriever로 사용할 수 있다.
# 이미 vector_db 폴더가 존재하는 경우, 기존 데이터를 삭제하고 새로 생성한다. 따라서 기존 데이터는 모두 사라지게 된다.
def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")

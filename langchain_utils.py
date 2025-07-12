import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.ensemble import EnsembleRetriever


# def preprocess_data(filepath="data/Reviews.xlsx"):
#     df = pd.read_excel(filepath)
#     df = df.dropna(subset=["Student Name"])
#     df = df[df["Student Name"].apply(lambda x: all(ord(char) < 128 for char in str(x)))]
#     df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
#     df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
#     df["Normalized Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
#     df = df[["Course Name", "Student Name", "Normalized Timestamp", "Rating", "Comment"]]
#     df["Comment"] = df["Comment"].fillna("No Comment")
#     df = df.drop_duplicates(subset=["Student Name", "Course Name"])

#     return df


def create_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = f"""
Course Name: {row['Course Name']}
Student Name: {row['Student Name']}
Timestamp: {row['Normalized Timestamp']}
Rating: {row['Rating']}
Comment: {row['Comment']}
"""
        documents.append(Document(page_content=content))
    return documents


def build_retrievers(documents):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    faiss_index = FAISS.from_documents(documents, embedding_model)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    store_hybrid = InMemoryStore()

    retriever_hybrid = ParentDocumentRetriever(
        vectorstore=FAISS.from_documents(documents, embedding_model),
        docstore=store_hybrid,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever_hybrid.add_documents(documents)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_index.as_retriever(), retriever_hybrid],
        weights=[0.6, 0.4]
    )
    return ensemble_retriever


def preprocess_data_from_df(df):
    """
    Process data from uploaded DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()    
    df = df.dropna(subset=["Student Name"])
    df = df[df["Student Name"].apply(lambda x: all(ord(char) < 128 for char in str(x)))]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    df["Normalized Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df = df[["Course Name", "Student Name", "Normalized Timestamp", "Rating", "Comment"]]
    df["Comment"] = df["Comment"].fillna("No Comment")
    df = df.drop_duplicates(subset=["Student Name", "Course Name"])
    
    # Select and clean final columns
    df = df[["Course Name", "Student Name", "Normalized Timestamp", "Rating", "Comment"]]
    df["Comment"] = df["Comment"].fillna("No Comment")
    df = df.drop_duplicates(subset=["Student Name", "Course Name"])

    return df

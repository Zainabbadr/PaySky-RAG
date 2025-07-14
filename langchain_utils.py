import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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


def preprocess_data_from_df(df):
    """
    Process data from uploaded DataFrame - works with any tabular data
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Basic data cleaning
    df = df.dropna(how='all')  # Remove completely empty rows
    df = df.fillna("N/A")  # Fill missing values with "N/A"
    
    # Convert all columns to string to ensure consistency
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # If there are datetime columns, try to normalize them
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass  # Keep as string if conversion fails
    
    return df


def create_documents(df):
    """
    Create documents from any DataFrame structure
    """
    documents = []
    for index, row in df.iterrows():
        # Create content by combining all column-value pairs
        content_parts = []
        metadata = {}
        
        for col in df.columns:
            value = row[col]
            content_parts.append(f"{col}: {value}")
            metadata[col.lower().replace(' ', '_')] = value
        
        content = "\n".join(content_parts)
        
        # Add row index for unique identification
        metadata['row_index'] = index
        
        documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    return documents


def build_retrievers(documents):
    """
    Build a simple, efficient retriever using FAISS
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create FAISS vector store
    faiss_index = FAISS.from_documents(documents, embedding_model)
    
    # Return the retriever
    return faiss_index.as_retriever(search_kwargs={"k": 5})


# Remove the old preprocess_data_from_df function as it's been replaced above

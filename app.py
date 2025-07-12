import streamlit as st
import pandas as pd
import tempfile
import os
import warnings

# Suppress torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from langchain_utils import preprocess_data_from_df, create_documents, build_retrievers
from groq_client import get_groq_client, ask_groq

st.set_page_config(page_title="PaySky HR Analytics Assistant", page_icon="📊", layout="wide")

st.title("PaySky HR Analytics Assistant")
st.markdown("Upload your Excel file with employee training records and ask questions about the data!")

# Function to build retriever with caching
@st.cache_resource
def build_cached_retriever(_documents):
    """Build retriever with caching to avoid rebuilding on every run"""
    return build_retrievers(_documents)

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing employee training records"
    )
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        st.json(file_details)

# Main content area
if uploaded_file is not None:
    try:
        # Load data from uploaded file
        with st.spinner("Loading and processing data..."):
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Show basic info about the data
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Students", df['Student Name'].nunique() if 'Student Name' in df.columns else 0)
            
            # Show column names
            st.write("**Columns in your data:**", list(df.columns))
            
            # Show sample data
            with st.expander("Preview Data (First 5 rows)"):
                st.dataframe(df.head())
            
            # Process data for RAG
            with st.spinner("Processing data..."):
                processed_df = preprocess_data_from_df(df)
                documents = create_documents(processed_df)
                st.success(f"Created {len(documents)} documents from your data")
            
            # Build retriever (cached)
            retriever = build_cached_retriever(documents)
            
            st.success("Search index built successfully! You can now ask questions.")
            
        # Query interface
        st.subheader("Ask Questions")
        
        # Sample questions
        sample_questions = [
            "What are the most common feedback comments?",
            "Which courses have the highest ratings?",
            "What do students say about the instructors?",
            "Are there any negative reviews I should be aware of?"
        ]
        
        # Dropdown for sample questions
        selected_sample = st.selectbox(
            "Choose a sample question or type your own:",
            [""] + sample_questions
        )
        
        # Text input for custom question
        if selected_sample:
            query = st.text_input("Your question:", value=selected_sample)
        else:
            query = st.text_input("Your question:", placeholder="Ask anything about the training records...")
        
        # Search button
        if st.button("Search", type="primary"):
            if query.strip():
                try:
                    with st.spinner("Searching relevant documents..."):
                        results = retriever.invoke(query)
                        context = "\n\n".join([doc.page_content for doc in results])
                    
                    # Show retrieved context
                    with st.expander("Retrieved Context"):
                        st.text_area("Context from documents", context, height=200)
                    
                    # Get answer from Groq
                    try:
                        client = get_groq_client()
                        with st.spinner("Generating answer..."):
                            answer = ask_groq(client, context, query)
                        
                        # Display answer
                        st.subheader("Answer")
                        st.write(answer)
                        
                    except Exception as e:
                        st.error(f"Error getting answer from Groq: {e}")
                        st.info("Please check your GROQ_API_KEY in the .env file")
                        
                except Exception as e:
                    st.error(f"Error during search: {e}")
            else:
                st.warning("Please enter a question!")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please make sure your Excel file has the expected columns: Course Name, Student Name, Timestamp, Rating, Comment")

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an Excel file to get started!")
    
    st.subheader("Expected File Format")
    st.markdown("""
    Your Excel file should contain columns like:
    - **Course Name**: Name of the training course
    - **Student Name**: Name of the student
    - **Timestamp**: When the feedback was given
    - **Rating**: Numeric rating (1-5)
    - **Comment**: Text feedback from students
    """)
    
    # Sample data format
    sample_data = {
        'Course Name': ['Python Programming', 'Data Science', 'Web Development'],
        'Student Name': ['John Smith', 'Jane Doe', 'Mike Johnson'],
        'Timestamp': ['2024-01-15 10:30', '2024-01-16 14:20', '2024-01-17 09:15'],
        'Rating': [5, 4, 5],
        'Comment': ['Great course!', 'Very informative', 'Excellent instructor']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.subheader("Sample Data Format")
    st.dataframe(sample_df)
    
    st.info("**Tip**: Make sure your Excel file follows a similar structure for best results!")

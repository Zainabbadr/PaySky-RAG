import streamlit as st
import pandas as pd
import tempfile
import os
import warnings

# Suppress torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from langchain_utils import preprocess_data_from_df, create_documents, build_retrievers
from groq_client import get_groq_client, ask_groq

st.set_page_config(page_title="PaySky Data Analytics Assistant", page_icon="📊", layout="wide")

st.title("PaySky Data Analytics Assistant")
st.markdown("Upload your Excel file with tabular data and ask questions about it using natural language!")

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
        help="Upload an Excel file containing your tabular data"
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
                # Try to find a column that might represent entities (people, items, etc.)
                entity_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['name', 'id', 'user', 'customer', 'student', 'employee', 'person', 'item', 'product']):
                        entity_col = col
                        break
                
                if entity_col:
                    st.metric("Unique Entities", df[entity_col].nunique())
                else:
                    st.metric("Data Points", len(df))
            
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
        
        # Sample questions - make them more generic
        sample_questions = [
            "What are the key insights from this data?",
            "What are the most common patterns or trends?",
            "What are the highest and lowest values in the data?",
            "Are there any outliers or unusual entries?",
            "What relationships can you identify between different columns?",
            "Can you summarize the main findings from this dataset?"
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
            query = st.text_input("Your question:", placeholder="Ask anything about your data...")
        
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
        st.info("Please make sure your Excel file is properly formatted with clear column headers")

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an Excel file to get started!")
    
    st.subheader("How it Works")
    st.markdown("""
    1. **Upload your Excel file** - Any tabular data with clear column headers
    2. **Data Processing** - The system will automatically analyze your data structure
    3. **Ask Questions** - Use natural language to query your data
    4. **Get Insights** - Receive AI-powered analysis and answers
    """)
    
    st.subheader("Supported Data Types")
    st.markdown("""
    This tool works with any tabular data including:
    - **Business Data**: Sales records, customer data, financial reports
    - **Survey Data**: Questionnaire responses, feedback forms
    - **Academic Data**: Student records, course evaluations, research data
    - **Operational Data**: Inventory, logistics, performance metrics
    - **And much more!**
    """)
    
    # Sample data format - make it more generic
    sample_data = {
        'Product': ['Laptop', 'Phone', 'Tablet'],
        'Category': ['Electronics', 'Electronics', 'Electronics'],
        'Price': [1200, 800, 600],
        'Rating': [4.5, 4.2, 4.0],
        'Reviews': ['Great performance', 'Good value', 'Compact design']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.subheader("Sample Data Format")
    st.dataframe(sample_df)
    
    st.info("**Tip**: Make sure your Excel file has clear column headers for best results!")

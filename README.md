# PaySky HR Analytics Assistant - RAG Implementation

## Overview
A Streamlit-based HR Analytics Assistant that processes Excel files containing employee training records and enables natural language querying using RAG (Retrieval-Augmented Generation) technology.

## Architecture & Implementation

### Core Technologies Used:
- **Frontend**: Streamlit (Web Interface)
- **Backend**: Python with LangChain ecosystem
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **LLM**: Groq API with `llama-3.3-70b-versatile` model
- **Data Processing**: Pandas with openpyxl

## Implementation Details

### Phase 1: Data Extraction & Preprocessing
**File: `app.py` (Main Application)**

**Technologies Used:**
- **Streamlit**: Web interface with file upload widget
- **Pandas**: Excel file reading with `openpyxl` engine
- **File Validation**: Support for `.xlsx` and `.xls` formats

**Implementation Steps:**
1. **File Upload Interface**
   - Streamlit file uploader widget
   - File size display and validation
   - Support for Excel formats only

2. **Data Loading**
   - `pd.read_excel()` for Excel file processing
   - Automatic data type detection
   - Column structure analysis

3. **Data Overview Display**
   - Total records count
   - Column count and names
   - Unique students count
   - Data preview (first 5 rows)

### Phase 2: Data Cleaning & Transformation
**File: `langchain_utils.py`**

**Technologies Used:**
- **Pandas**: Data manipulation and cleaning
- **LangChain**: Document processing pipeline
- **Text Processing**: String manipulation and formatting

**Implementation Steps:**
1. **Data Preprocessing Function (`preprocess_data_from_df`)**
   - Implements flexible column mapping to handle various Excel file structures without requiring strict column names
   - Uses intelligent column detection by searching for keywords like 'course', 'student', 'rating', and 'comment' in column names
   - Gracefully handles missing columns by providing default values or skipping optional fields
   - Converts all timestamp data to string format to ensure consistent processing across different date formats
   - Applies systematic missing value handling by filling null values with "N/A" to maintain data integrity

2. **Adaptive Column Detection**
   - Employs case-insensitive keyword matching to identify relevant columns regardless of naming conventions
   - Provides fallback mechanisms when expected columns are not found in the uploaded file
   - Maintains flexibility to work with different Excel templates and formats used across organizations

3. **Data Structure Standardization**
   - Ensures consistent column naming conventions throughout the processing pipeline
   - Normalizes data types to prevent downstream processing errors
   - Implements comprehensive missing value handling strategies to maintain data quality

### Phase 3: Document Creation & Chunking
**File: `langchain_utils.py`**

**Technologies Used:**
- **LangChain Document**: Structured document creation
- **Custom Chunking**: Row-level chunking strategy
- **Metadata Preservation**: Rich metadata for each document

**Implementation Steps:**
1. **Document Creation Function (`create_documents`)**
   - Transforms each row of the Excel data into a separate LangChain Document object, enabling granular search capabilities
   - Embeds comprehensive metadata including course information, student details, timestamps, and ratings for enhanced retrieval accuracy
   - Formats page content in a structured, human-readable format that optimizes both search relevance and LLM comprehension
   - Maintains data relationships through consistent metadata schemas across all documents

2. **Row-Level Chunking Strategy**
   - Adopts a granular approach where each training record becomes an individual searchable unit
   - Preserves all original data relationships and context within each document chunk
   - Enables precise retrieval of specific training feedback and course evaluations
   - Maintains metadata integrity to support filtered and targeted search queries

3. **Document Structure Design**
   - Creates well-formatted document content that combines all relevant information for each training record
   - Implements a consistent metadata schema that supports various query types and filtering needs
   - Ensures optimal balance between document size and information completeness for effective vector search
   - Structures content to facilitate both semantic similarity matching and specific information extraction

### Phase 4: Embedding Generation & Vector Storage
**File: `langchain_utils.py`**

**Technologies Used:**
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` model
- **LangChain HuggingFace**: `langchain-huggingface` integration
- **FAISS**: Facebook AI Similarity Search for vector storage

**Implementation Steps:**
1. **Embedding Model Selection and Configuration**
   - Utilizes the all-MiniLM-L6-v2 model, which provides an optimal balance between embedding quality and computational efficiency
   - This sentence transformer model excels at capturing semantic meaning in short to medium-length texts, making it ideal for training feedback and course comments
   - Configured through LangChain's HuggingFace integration for seamless pipeline integration and consistent performance
   - The model generates 384-dimensional embeddings that capture nuanced semantic relationships in the training data

2. **Vector Store Architecture**
   - Implements FAISS (Facebook AI Similarity Search) for high-performance vector storage and retrieval operations
   - Uses a simplified, streamlined architecture that eliminates complex ensemble approaches in favor of direct efficiency
   - Establishes direct document-to-vector mapping that maintains clear traceability from search results back to original training records
   - Optimizes for local development and deployment scenarios while maintaining production-ready performance standards

3. **Retriever Building and Optimization**
   - Constructs FAISS vector store from processed documents with optimized indexing for fast similarity search
   - Configures retriever parameters to balance search accuracy with response speed for optimal user experience
   - Implements performance optimizations specifically tuned for HR analytics use cases and typical query patterns
   - Ensures consistent retrieval quality across different types of training feedback and course evaluation data

### Phase 5: Caching & Performance Optimization
**File: `app.py`**

**Technologies Used:**
- **Streamlit Caching**: `@st.cache_resource` decorator
- **Performance Optimization**: Cached retriever building
- **Memory Management**: Efficient resource utilization

**Implementation Steps:**
1. **Intelligent Caching Strategy**
   - Implements Streamlit's resource caching decorator to eliminate redundant retriever building operations across user sessions
   - The caching mechanism recognizes identical document sets and reuses previously built retrievers, dramatically reducing response times
   - Employs session-based caching that persists the expensive embedding and vector store operations until the user uploads new data
   - Optimizes memory usage by caching only the essential retriever components rather than raw data or intermediate processing steps

2. **Performance Optimization Benefits**
   - Reduces initial loading time from multiple seconds to near-instantaneous responses for subsequent queries
   - Eliminates repetitive computational overhead associated with embedding generation and vector index creation
   - Provides seamless user experience by running all caching operations silently in the background without UI disruption
   - Maintains system responsiveness even with large Excel files containing thousands of training records

3. **Memory and Resource Management**
   - Implements efficient resource utilization patterns that prevent memory leaks during long user sessions
   - Balances caching benefits with memory constraints by selectively caching only the most computationally expensive operations
   - Ensures optimal performance across different hardware configurations and deployment environments
   - Maintains stable performance characteristics regardless of file size or complexity within reasonable limits

### Phase 6: LLM Integration & Response Generation
**File: `groq_client.py`**

**Technologies Used:**
- **Groq API**: Cloud-based LLM service
- **Model**: `llama-3.3-70b-versatile` (Latest Llama model)
- **Environment Management**: `python-dotenv` for API key management
- **Streaming**: Real-time response generation

**Implementation Steps:**
1. **Advanced LLM Service Integration**
   - Integrates with Groq's cloud-based infrastructure to leverage the powerful Llama 3.3 70B model for high-quality natural language understanding and generation
   - Implements secure API key management through environment variables to protect sensitive credentials and enable flexible deployment across different environments
   - Utilizes Groq's optimized inference infrastructure for faster response times compared to traditional LLM hosting solutions
   - Provides robust error handling and graceful degradation when API services are unavailable or rate limits are exceeded

2. **Context-Aware Response Generation**
   - Develops sophisticated prompt engineering strategies that effectively combine retrieved training data with user queries for accurate, contextual responses
   - Implements streaming response generation that provides real-time feedback to users, improving perceived performance and engagement
   - Maintains conversation context awareness to enable follow-up questions and complex multi-turn interactions about the training data
   - Ensures response quality through careful prompt structuring that guides the LLM to provide relevant, accurate, and professionally formatted answers

3. **Prompt Engineering and Optimization**
   - Designs structured prompts that clearly delineate retrieved context from user questions, ensuring the LLM understands the HR analytics domain
   - Implements clear instructions for response formatting that maintains consistency across different types of queries and data scenarios
   - Optimizes prompt structure to maximize the effectiveness of the 70B parameter model while staying within context length limitations
   - Incorporates domain-specific guidance to ensure responses are appropriate for HR analytics use cases and maintain professional tone

### Phase 7: Search & Retrieval Pipeline
**File: `app.py`**

**Technologies Used:**
- **Semantic Search**: Vector similarity search via FAISS
- **LangChain Integration**: Seamless retriever integration
- **Context Assembly**: Retrieved document processing

**Implementation Steps:**
1. **Intelligent Search Interface Design**
   - Provides users with both pre-defined sample questions and custom query input options to accommodate different user preferences and expertise levels
   - Implements intuitive dropdown menus populated with HR analytics-specific sample questions that demonstrate the system's capabilities
   - Incorporates smart search button functionality with appropriate loading states to provide clear feedback during processing
   - Ensures seamless integration between user interface elements and underlying search functionality for optimal user experience

2. **Advanced Retrieval Processing**
   - Executes semantic vector similarity search using FAISS to identify the most relevant training records based on query context and meaning
   - Assembles retrieved documents into coherent context blocks that maintain logical relationships between related training feedback
   - Implements intelligent document ranking and selection to ensure the most pertinent information is included in the LLM context
   - Provides transparent access to retrieved context through expandable interface elements, allowing users to verify and understand the source of generated responses

3. **Context Processing and Optimization**
   - Processes retrieved documents to create well-structured context that maximizes LLM comprehension and response quality
   - Implements relevance scoring and filtering mechanisms to ensure only the most pertinent training data influences the generated responses
   - Maintains optimal balance between context richness and processing efficiency to deliver timely responses without sacrificing accuracy
   - Ensures consistent context formatting that enables the LLM to effectively distinguish between different training records and their associated metadata

### Phase 8: User Interface & Experience
**File: `app.py`**

**Technologies Used:**
- **Streamlit Components**: File uploader, metrics, expandable sections
- **Layout Management**: Sidebar navigation, column layouts
- **Interactive Elements**: Dropdown menus, text inputs, buttons

**Implementation Steps:**
1. **File Upload Section**
   - Sidebar file uploader
   - File validation and details display
   - Success/error feedback

2. **Data Overview Dashboard**
   - Metrics display (records, columns, students)
   - Column names listing
   - Data preview with expandable section

3. **Query Interface**
   - Pre-defined sample questions
   - Custom question input
   - Search results with context display
   - Streaming answer generation

4. **Error Handling & User Guidance**
   - Clear error messages
   - Usage instructions
   - Expected file format guidance
   - Sample data format display

## Project Structure

```
PaySky-RAG/
├── app.py                  # Main Streamlit application
├── langchain_utils.py      # Data processing & RAG utilities
├── groq_client.py         # Groq API integration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── data/
    └── Reviews.xlsx      # Sample training data
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Groq API key

### Step 1: Clone Repository
```bash
git clone https://github.com/Zainabbadr/PaySky-RAG.git
cd PaySky-RAG
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Run Application
```bash
streamlit run app.py
```

## Dependencies (requirements.txt)

```
streamlit==1.29.0
pandas==2.1.4
openpyxl==3.1.2
langchain-community==0.0.13
langchain-huggingface==0.0.1
langchain-core==0.1.12
faiss-cpu==1.7.4
sentence-transformers==2.2.2
groq==0.4.1
python-dotenv==1.0.0
```

## Features

### ✅ Implemented Features
- **Excel File Upload**: Support for .xlsx and .xls formats
- **Data Processing**: Flexible column mapping and data cleaning
- **Document Creation**: Row-level chunking with metadata preservation
- **Vector Search**: FAISS-based similarity search
- **LLM Integration**: Groq API with Llama 3.3 70B model
- **Caching**: Optimized performance with Streamlit caching
- **User Interface**: Intuitive Streamlit web interface
- **Error Handling**: Comprehensive error management
- **Sample Questions**: Pre-defined HR analytics queries

### 🔍 Sample Questions
- "What are the most common feedback comments?"
- "Which courses have the highest ratings?"
- "What do students say about the instructors?"
- "Are there any negative reviews I should be aware of?"

### 📊 Expected Data Format
The application expects Excel files with columns like:
- **Course Name**: Name of the training course
- **Student Name**: Name of the student
- **Timestamp**: When the feedback was given
- **Rating**: Numeric rating (1-5)
- **Comment**: Text feedback from students

## Architecture Benefits

### Performance Optimizations
- **Caching**: Retriever built only once per session
- **Efficient Embeddings**: Lightweight all-MiniLM-L6-v2 model
- **FAISS Integration**: Fast similarity search
- **Streaming Responses**: Real-time LLM output

### Scalability Features
- **Modular Design**: Separate utilities for different functions
- **Flexible Data Processing**: Handles various Excel structures
- **Error Resilience**: Graceful handling of missing data
- **Memory Efficient**: Optimized resource usage

## Technical Decisions

### Model Selection Rationale
- **Embedding Model**: `all-MiniLM-L6-v2` chosen for balance of performance and speed
- **Vector Database**: FAISS selected for simplicity and local development
- **LLM**: Groq with Llama 3.3 70B for high-quality responses with good speed
- **Framework**: LangChain for RAG pipeline standardization

### Architecture Choices
- **Streamlit**: Rapid prototyping and intuitive UI
- **Row-level Chunking**: Granular retrieval for specific queries
- **Simplified Pipeline**: Removed complex ensemble retrieval for better performance
- **Caching Strategy**: Session-based caching for optimal UX

## Future Enhancements

### Potential Improvements
- **Multi-sheet Support**: Process multiple Excel sheets
- **Advanced Analytics**: Statistical analysis and visualizations
- **User Authentication**: Secure access and user management
- **Export Features**: Download results and reports
- **Database Integration**: Persistent storage of processed data
- **API Endpoints**: RESTful API for external integration

### Deployment Options
- **Cloud Deployment**: AWS, Azure, or GCP hosting
- **Containerization**: Docker for consistent deployment
- **Load Balancing**: Handle multiple concurrent users
- **Monitoring**: Performance and usage analytics

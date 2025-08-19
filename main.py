import os
import logging
import asyncio
import threading
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session


# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from werkzeug.utils import secure_filename
import tempfile
import json
import sys
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python version info for debugging
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for the AI system
qa_chain = None
vectorstore = None
system_initialized = False

# Enhanced legal document templates
DOCUMENT_TEMPLATES = {
    'civil': {
        'title': 'Civil Suit',
        'court_type': 'Civil Court',
        'format': 'standard_civil'
    },
    'criminal': {
        'title': 'Criminal Complaint',
        'court_type': 'Magistrate Court',
        'format': 'criminal_complaint'
    },
    'constitutional': {
        'title': 'Constitutional Petition',
        'court_type': 'High Court/Supreme Court',
        'format': 'writ_petition'
    },
    'writ_petition': {
        'title': 'Writ Petition',
        'court_type': 'High Court',
        'format': 'writ_petition'
    }
}

def run_async_in_thread(coro):
    """
    Run an async function in a new thread with its own event loop.
    This solves the event loop issue in Flask threads.
    """
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def ensure_event_loop():
    """
    Ensure there's an event loop in the current thread.
    Alternative approach - simpler but less robust.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_enhanced_prompt():
    """Create an enhanced prompt template for legal document generation"""
    
    prompt_template = """
    You are an expert legal assistant specializing in Indian law and the Indian Constitution. 
    Your task is to generate a comprehensive, professionally formatted legal document that can be filed in Indian courts.

    CONTEXT FROM INDIAN CONSTITUTION:
    {context}

    USER'S CASE DETAILS:
    {question}

    INSTRUCTIONS:
    Please generate a complete legal document with the following structure:

    1. CASE TITLE AND COURT DETAILS:
       - Proper case title format
       - Appropriate court jurisdiction
       - Case number placeholder

    2. PARTIES TO THE CASE:
       - Complete details of plaintiff/petitioner
       - Complete details of defendant/respondent

    3. JURISDICTION AND VENUE:
       - Legal basis for court's jurisdiction
       - Proper venue justification

    4. STATEMENT OF FACTS:
       - Chronological presentation of facts
       - Relevant dates and circumstances
       - Supporting evidence references

    5. LEGAL GROUNDS AND CAUSES OF ACTION:
       - Relevant constitutional articles
       - Applicable laws and statutes
       - Legal precedents (where applicable)
       - Rights violated or legal issues

    6. PRAYER/RELIEF SOUGHT:
       - Specific remedies requested
       - Alternative reliefs
       - Costs and other claims

    7. LEGAL FORMATTING:
       - Proper legal language and terminology
       - Numbered paragraphs
       - Appropriate legal citations
       - Professional document structure

    8. VERIFICATION CLAUSE:
       - Standard verification format
       - Signature line for petitioner/advocate

    REQUIREMENTS:
    - Use formal legal language appropriate for Indian courts
    - Include relevant constitutional provisions and legal citations
    - Ensure the document is court-ready and professionally formatted
    - Base arguments on solid constitutional and legal grounds
    - Include proper legal terminology and phrases used in Indian legal practice

    GENERATED LEGAL DOCUMENT:
    """
    
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

def initialize_system():
    """Initialize the LangChain system with enhanced error handling and event loop management"""
    global qa_chain, vectorstore, system_initialized
    
    try:
        logger.info("Starting system initialization...")
        
        # Ensure event loop exists - Method 1 (Simple)
        ensure_event_loop()
        
        # Check for API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Check if PDF exists
        pdf_path = 'indian_constitution.pdf'
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Constitution PDF not found at {pdf_path}")
        
        # Load the Indian Constitution PDF
        logger.info("Loading Indian Constitution PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found in the PDF")
        
        logger.info(f"Loaded {len(documents)} pages from the Constitution")
        
        # Split documents into chunks with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Split into {len(texts)} text chunks")
        
        # Create embeddings and vector store with error handling
        logger.info("Creating embeddings...")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(texts, embeddings)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            # Fallback: try with different approach or show specific error
            if "event loop" in str(e).lower():
                logger.info("Retrying with thread-safe approach...")
                # You could implement a retry with the thread-based approach here
                raise RuntimeError("Event loop issue detected. Please restart the application and try again.")
            raise
        
        # Initialize the LLM with optimized parameters
        llm = GoogleGenerativeAI(
            model='gemini-2.0-flash',
            temperature=0.2,  # Lower for more consistent legal documents
            max_output_tokens=8192
        )
        
        # Create enhanced prompt template
        prompt = create_enhanced_prompt()
        
        # Create the QA chain with enhanced retrieval
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 8,  # Retrieve more relevant chunks
                    "fetch_k": 20,  # Consider more documents for MMR
                    "lambda_mult": 0.5  # Diversity parameter
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        system_initialized = True
        logger.info("System initialization completed successfully!")
        return "System initialized successfully! Ready to generate legal documents."
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        system_initialized = False
        raise e

# Alternative initialization function using thread-based approach
def initialize_system_threaded():
    """Initialize the system using a separate thread with its own event loop"""
    global qa_chain, vectorstore, system_initialized
    
    async def async_init():
        try:
            logger.info("Starting system initialization in async context...")
            
            # Check for API key
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            # Check if PDF exists
            pdf_path = 'indian_constitution.pdf'
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Constitution PDF not found at {pdf_path}")
            
            # Load the Indian Constitution PDF
            logger.info("Loading Indian Constitution PDF...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in the PDF")
            
            logger.info(f"Loaded {len(documents)} pages from the Constitution")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            logger.info(f"Split into {len(texts)} text chunks")
            
            # Create embeddings and vector store
            logger.info("Creating embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # Initialize the LLM
            llm = GoogleGenerativeAI(
                model='gemini-2.0-flash',
                temperature=0.2,
                max_output_tokens=8192
            )
            
            # Create enhanced prompt template
            prompt = create_enhanced_prompt()
            
            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 8,
                        "fetch_k": 20,
                        "lambda_mult": 0.5
                    }
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            return "System initialized successfully! Ready to generate legal documents."
            
        except Exception as e:
            logger.error(f"Async initialization failed: {str(e)}")
            raise e
    
    try:
        # Run the async initialization in a separate thread
        result = run_async_in_thread(async_init())
        system_initialized = True
        return result
    except Exception as e:
        system_initialized = False
        raise e

@app.route('/')
def home():
    """Home page with enhanced form"""
    return render_template('index.html', system_status=system_initialized)

@app.route('/generate_document', methods=['POST'])
def generate_document():
    """Generate legal document with enhanced processing"""
    try:
        # Ensure event loop for this request
        ensure_event_loop()
        
        if not system_initialized:
            return jsonify({
                "error": "System not initialized. Please upload Constitution PDF and initialize the system first."
            }), 500
        
        # Extract and validate form data
        case_type = request.form.get('case_type', '').strip()
        plaintiff_name = request.form.get('plaintiff_name', '').strip()
        defendant_name = request.form.get('defendant_name', '').strip()
        case_details = request.form.get('case_details', '').strip()
        relief_sought = request.form.get('relief_sought', '').strip()
        
        # Validation
        if not all([case_type, plaintiff_name, defendant_name, case_details, relief_sought]):
            return jsonify({"error": "All fields are required"}), 400
        
        # Get template information
        template_info = DOCUMENT_TEMPLATES.get(case_type, {
            'title': 'Legal Case',
            'court_type': 'Appropriate Court',
            'format': 'standard'
        })
        
        # Construct enhanced query
        user_query = f"""
        CASE TYPE: {template_info['title']}
        COURT TYPE: {template_info['court_type']}
        
        PARTIES:
        Plaintiff/Petitioner: {plaintiff_name}
        Defendant/Respondent: {defendant_name}
        
        CASE DETAILS AND FACTS:
        {case_details}
        
        RELIEF/REMEDY SOUGHT:
        {relief_sought}
        
        Please generate a complete, court-ready legal document based on these details and relevant provisions of the Indian Constitution. The document should be professionally formatted and include all necessary legal elements for filing in {template_info['court_type']}.
        """
        
        logger.info(f"Generating legal document for case type: {case_type}")
        
        # Generate the document using the QA chain
        result = qa_chain({"query": user_query})
        
        legal_document = result['result']
        source_docs = result['source_documents']
        
        # Process constitutional references
        constitutional_refs = []
        for i, doc in enumerate(source_docs[:5]):  # Top 5 most relevant documents
            constitutional_refs.append({
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'source': f"Indian Constitution - Page {doc.metadata.get('page', i+1)}",
                'relevance_score': f"Reference {i+1}"
            })
        
        # Log successful generation
        logger.info("Legal document generated successfully")
        
        # Store generation info in session for analytics
        session['last_generation'] = {
            'timestamp': datetime.now().isoformat(),
            'case_type': case_type,
            'status': 'success'
        }
        
        return render_template('result.html',
                             legal_document=legal_document,
                             constitutional_refs=constitutional_refs,
                             case_details={
                                 'case_type': case_type,
                                 'plaintiff': plaintiff_name,
                                 'defendant': defendant_name,
                                 'template_info': template_info
                             },
                             generation_timestamp=datetime.now().strftime("%B %d, %Y at %I:%M %p"))
        
    except Exception as e:
        logger.error(f"Document generation error: {str(e)}")
        return jsonify({"error": f"Document generation failed: {str(e)}"}), 500

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the system with enhanced error handling"""
    try:
        # Try the simple approach first
        try:
            message = initialize_system()
        except Exception as e:
            if "event loop" in str(e).lower():
                logger.info("Falling back to threaded initialization...")
                message = initialize_system_threaded()
            else:
                raise e
        
        return jsonify({"message": message, "status": "success"})
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return jsonify({"error": f"Initialization failed: {str(e)}", "status": "error"}), 500

@app.route('/upload_constitution', methods=['POST'])
def upload_constitution():
    """Upload Indian Constitution PDF with enhanced validation"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename("indian_constitution.pdf")
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save uploaded file
            file.save(file_path)
            
            # Move to main directory
            final_path = "indian_constitution.pdf"
            if os.path.exists(final_path):
                os.remove(final_path)  # Remove old file
            
            os.rename(file_path, final_path)
            
            logger.info("Constitution PDF uploaded successfully")
            
            # Auto-initialize the system with error handling
            try:
                message = initialize_system()
            except Exception as e:
                if "event loop" in str(e).lower():
                    logger.info("Using threaded initialization after upload...")
                    message = initialize_system_threaded()
                else:
                    raise e
            
            return jsonify({
                "message": f"Constitution uploaded and {message}",
                "status": "success"
            })
        else:
            return jsonify({"error": "Please upload a valid PDF file"}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/system_status')
def system_status():
    """Check system status"""
    return jsonify({
        "initialized": system_initialized,
        "constitution_pdf_exists": os.path.exists("indian_constitution.pdf"),
        "api_key_set": bool(os.environ.get("GOOGLE_API_KEY")),
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_message="Internal server error"), 500

if __name__ == '__main__':
    print("="*60)
    print("🏛️  LEGAL DOCUMENT GENERATOR")
    print("="*60)
    print("🔧 Starting application...")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🔑 API Key set: {'✅' if os.environ.get('GOOGLE_API_KEY') else '❌'}")
    print(f"📄 Constitution PDF: {'✅' if os.path.exists('indian_constitution.pdf') else '❌'}")
    print("="*60)
    print("📋 SETUP CHECKLIST:")
    print("1. Set GOOGLE_API_KEY environment variable")
    print("2. Upload Indian Constitution PDF")
    print("3. Initialize the system")
    print("4. Start generating legal documents!")
    print("="*60)
    
    # Use PORT from environment for Render deployment
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🌐 Server starting at http://{host}:{port}")
    print("="*60)
    
    app.run(debug=False, host=host, port=port)  # Set debug=False for production
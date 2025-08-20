import os
import logging
import asyncio
import threading
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session

# Set event loop policy for Windows compatibility
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Updated LangChain imports - replacing FAISS with ChromaDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Changed from FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from werkzeug.utils import secure_filename
import tempfile
import json
import sys
import platform
import traceback

# Print Python version info for debugging
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates' if os.path.exists('templates') else '.')
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
initialization_error = None

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

def run_in_thread_with_loop(func, *args, **kwargs):
    """Run a function in a thread with its own event loop"""
    import concurrent.futures
    import threading
    
    result_container = {}
    exception_container = {}
    
    def thread_target():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = func(*args, **kwargs)
                result_container['result'] = result
            finally:
                loop.close()
        except Exception as e:
            exception_container['exception'] = e
    
    # Run in a separate thread
    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()
    
    # Check for exceptions
    if 'exception' in exception_container:
        raise exception_container['exception']
    
    return result_container.get('result')

def initialize_system_sync():
    """Synchronous version of system initialization"""
    global qa_chain, vectorstore, system_initialized, initialization_error
    
    logger.info("Starting synchronous system initialization...")
    initialization_error = None
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set your Google API key.")
    
    # Check if PDF exists
    pdf_path = 'indian_constitution.pdf'
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Constitution PDF not found at {pdf_path}. Please upload the Indian Constitution PDF first.")
    
    # Load the Indian Constitution PDF
    logger.info("Loading Indian Constitution PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No content found in the PDF. Please ensure the PDF contains readable text.")
    
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
    
    # Create embeddings - this is the problematic part, so we'll handle it carefully
    logger.info("Creating embeddings with ChromaDB...")
    
    # Initialize embeddings with explicit API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create ChromaDB vectorstore
    if os.environ.get('RENDER'):  # Render deployment
        import tempfile
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {temp_dir}")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=temp_dir
        )
    else:  # Local development
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    logger.info("ChromaDB vectorstore created successfully")
    
    # Initialize the LLM
    try:
        llm = GoogleGenerativeAI(
            model='gemini-1.5-flash',  # Use more stable model
            temperature=0.2,
            max_output_tokens=8192,
            google_api_key=api_key
        )
        logger.info("LLM initialized successfully with gemini-1.5-flash")
    except Exception as e:
        logger.warning(f"Failed to initialize gemini-1.5-flash: {e}, trying gemini-pro")
        llm = GoogleGenerativeAI(
            model='gemini-pro',
            temperature=0.2,
            max_output_tokens=8192,
            google_api_key=api_key
        )
        logger.info("LLM initialized with fallback model gemini-pro")
    
    # Create enhanced prompt template
    prompt = create_enhanced_prompt()
    logger.info("Prompt template created successfully")
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    logger.info("QA chain created successfully")
    
    return "System initialized successfully! Ready to generate legal documents."

def initialize_system():
    """Initialize the LangChain system with enhanced error handling and event loop management"""
    global qa_chain, vectorstore, system_initialized, initialization_error
    
    try:
        logger.info("Starting system initialization...")
        initialization_error = None
        
        # Try to run initialization in a thread with its own event loop
        try:
            message = run_in_thread_with_loop(initialize_system_sync)
            system_initialized = True
            logger.info("System initialization completed successfully!")
            return message
        except Exception as thread_e:
            logger.warning(f"Thread-based initialization failed: {thread_e}")
            logger.info("Attempting direct synchronous initialization...")
            
            # Fallback: try direct synchronous initialization
            message = initialize_system_sync()
            system_initialized = True
            logger.info("Direct initialization completed successfully!")
            return message
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"System initialization failed: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        system_initialized = False
        initialization_error = error_msg
        raise e

@app.route('/')
def home():
    """Home page with enhanced form"""
    try:
        return render_template('index.html', system_status=system_initialized)
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        # Fallback to basic HTML if template not found
        return """
        <!DOCTYPE html>
        <html><head><title>Legal Document Generator</title></head>
        <body style="font-family: Arial, sans-serif; margin: 50px;">
        <h1>Legal Document Generator</h1>
        <p>System is starting up. Please refresh the page in a moment.</p>
        <p><a href="/">Refresh Page</a></p>
        </body></html>
        """

@app.route('/generate_document', methods=['POST'])
def generate_document():
    """Generate legal document with enhanced processing"""
    try:
        if not system_initialized:
            error_msg = initialization_error or "System not initialized. Please upload Constitution PDF and initialize the system first."
            return jsonify({"error": error_msg}), 500
        
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
        try:
            result = qa_chain({"query": user_query})
            legal_document = result['result']
            source_docs = result['source_documents']
        except Exception as e:
            logger.error(f"Document generation error: {str(e)}")
            return jsonify({"error": f"Failed to generate document: {str(e)}"}), 500
        
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
        logger.info("Received initialization request")
        message = initialize_system()
        return jsonify({
            "message": message, 
            "status": "success",
            "initialized": system_initialized
        })
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Initialization error: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            "error": error_msg, 
            "status": "error",
            "initialized": False,
            "details": "Check server logs for more information"
        }), 500

@app.route('/upload_constitution', methods=['POST'])
def upload_constitution():
    """Upload Indian Constitution PDF with enhanced validation"""
    try:
        logger.info("Received file upload request")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "status": "error"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected", "status": "error"}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a valid PDF file", "status": "error"}), 400
        
        try:
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
            
            return jsonify({
                "message": "Constitution PDF uploaded successfully. Please click 'Initialize System' to complete setup.",
                "status": "success",
                "next_step": "Click 'Initialize System' button"
            })
            
        except Exception as e:
            logger.error(f"File handling error: {str(e)}")
            return jsonify({
                "error": f"Failed to save file: {str(e)}", 
                "status": "error"
            }), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            "error": f"Upload failed: {str(e)}", 
            "status": "error"
        }), 500

@app.route('/system_status')
def system_status():
    """Check system status"""
    return jsonify({
        "initialized": system_initialized,
        "constitution_pdf_exists": os.path.exists("indian_constitution.pdf"),
        "api_key_set": bool(os.environ.get("GOOGLE_API_KEY")),
        "timestamp": datetime.now().isoformat(),
        "initialization_error": initialization_error
    })

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Endpoint not found", "status": "error"}), 404
    
    # Try to render template, fallback to simple HTML if not found
    try:
        return render_template('error.html', error_message="Page not found"), 404
    except:
        return """
        <!DOCTYPE html>
        <html><head><title>404 - Page Not Found</title></head>
        <body style="font-family: Arial, sans-serif; margin: 50px; text-align: center;">
        <h1>404 - Page Not Found</h1>
        <p>The page you're looking for doesn't exist.</p>
        <p><a href="/">Go Back Home</a></p>
        </body></html>
        """, 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    if request.path.startswith('/api/') or request.is_json or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({"error": "Internal server error", "status": "error"}), 500
    
    # Try to render template, fallback to simple HTML if not found
    try:
        return render_template('error.html', error_message="Internal server error"), 500
    except:
        return """
        <!DOCTYPE html>
        <html><head><title>500 - Internal Server Error</title></head>
        <body style="font-family: Arial, sans-serif; margin: 50px; text-align: center;">
        <h1>500 - Internal Server Error</h1>
        <p>Something went wrong on our end. Please try again later.</p>
        <p><a href="/">Go Back Home</a></p>
        </body></html>
        """, 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    
    if request.path.startswith('/api/') or request.is_json or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({
            "error": "An unexpected error occurred", 
            "status": "error",
            "details": str(e)
        }), 500
    
    # Try to render template, fallback to simple HTML if not found
    try:
        return render_template('error.html', error_message=f"An unexpected error occurred: {str(e)}"), 500
    except:
        return """
        <!DOCTYPE html>
        <html><head><title>Error</title></head>
        <body style="font-family: Arial, sans-serif; margin: 50px; text-align: center;">
        <h1>An Error Occurred</h1>
        <p>Something went wrong. Please try again later.</p>
        <p><a href="/">Go Back Home</a></p>
        </body></html>
        """, 500

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
    
    app.run(debug=False, host=host, port=port)
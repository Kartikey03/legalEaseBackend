# Legal Document Generator 🏛️

An AI-powered legal document generator that creates professional, court-ready legal documents based on the Indian Constitution and legal precedents. This application uses advanced natural language processing to analyze constitutional provisions and generate comprehensive legal documents for various case types.

## 🌟 Features

- **AI-Powered Document Generation**: Uses Google's Gemini AI to create professional legal documents
- **Constitutional Knowledge Base**: Leverages the Indian Constitution as a primary reference source
- **Multiple Document Types**: Supports various legal document formats including:
  - Civil Suits
  - Criminal Complaints
  - Constitutional Petitions
  - Writ Petitions
- **Court-Ready Format**: Generates documents with proper legal formatting and structure
- **Constitutional References**: Provides relevant constitutional provisions and legal citations
- **User-Friendly Interface**: Simple web interface for easy document generation

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **AI/ML**: LangChain with Google Generative AI (Gemini)
- **Vector Database**: ChromaDB for document embeddings
- **PDF Processing**: PyPDF for constitution text extraction
- **Frontend**: HTML/CSS with Bootstrap styling

## 📋 Prerequisites

Before running this application, ensure you have:

- Python 3.8 or higher
- Google AI API Key (from Google AI Studio)
- Indian Constitution PDF file

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legal-document-generator.git
cd legal-document-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_ai_api_key_here
```

Or set the environment variable directly:

```bash
export GOOGLE_API_KEY="your_google_ai_api_key_here"
```

### 4. Get Your Google AI API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and add it to your environment variables

### 5. Run the Application

```bash
python main.py
```

The application will start at `http://localhost:5000`

## 📖 Usage Guide

### Initial Setup

1. **Upload Constitution PDF**: 
   - Click "Choose File" and select your Indian Constitution PDF
   - Click "Upload Constitution PDF"

2. **Initialize System**:
   - Click "Initialize System" to process the constitution and set up the AI model
   - Wait for the "System initialized successfully!" message

### Generating Legal Documents

1. **Select Document Type**: Choose from Civil, Criminal, Constitutional, or Writ Petition
2. **Fill Case Details**:
   - Plaintiff/Petitioner Name
   - Defendant/Respondent Name  
   - Detailed case description
   - Relief/remedy sought

3. **Generate Document**: Click "Generate Legal Document"
4. **Review Output**: The system will generate a complete legal document with:
   - Proper case formatting
   - Constitutional references
   - Legal citations
   - Professional structure

## 🏗️ Project Structure

```
legal-document-generator/
├── main.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/             # HTML templates (if using)
│   ├── index.html
│   ├── result.html
│   └── error.html
├── uploads/               # Temporary file uploads
├── chroma_db/            # ChromaDB vector database
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google AI API key (required)
- `PORT`: Server port (default: 5000)

### Application Settings

- `MAX_FILE_SIZE`: Maximum PDF upload size (50MB)
- `ALLOWED_EXTENSIONS`: Allowed file types (PDF only)
- `UPLOAD_FOLDER`: Directory for file uploads

## 🚀 Deployment

### Local Development

```bash
python main.py
```

### Production Deployment (using Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

### Render.com Deployment

This application is configured for easy deployment on Render.com:

1. Connect your GitHub repository to Render
2. Set the environment variable `GOOGLE_API_KEY`
3. The app will automatically deploy using the included configuration

## 📝 Document Types Supported

### Civil Suits
- Property disputes
- Contract violations
- Tort claims
- Family law matters

### Criminal Complaints
- First Information Reports (FIR)
- Criminal complaints under IPC
- Special acts violations

### Constitutional Petitions
- Fundamental rights violations
- Constitutional challenges
- Public interest litigation

### Writ Petitions
- Habeas Corpus
- Mandamus
- Prohibition
- Certiorari

## 🛡️ Legal Disclaimer

**Important**: This application is designed to assist in legal document preparation and should not replace professional legal advice. Always consult with qualified legal professionals before filing any legal documents in court.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.8+
- Flask 3.0.3
- LangChain 0.1.20
- Google Generative AI
- ChromaDB 0.4.22
- PyPDF 4.3.1

See `requirements.txt` for complete dependency list.

## 🐛 Troubleshooting

### Common Issues

**"GOOGLE_API_KEY environment variable not set"**
- Ensure you have set your Google AI API key in environment variables

**"Constitution PDF not found"**
- Upload a valid Indian Constitution PDF file through the web interface

**"System initialization failed"**
- Check your internet connection
- Verify your Google AI API key is valid
- Ensure the PDF file is readable and not corrupted

**Memory Issues**
- The application processes large documents; ensure sufficient RAM (minimum 2GB recommended)

### Logs

Check the console output for detailed error messages and debugging information.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Generative AI for powering the document generation
- LangChain for the AI framework
- ChromaDB for vector storage capabilities
- The Indian legal system for document format standards

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/legal-document-generator/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Support for additional regional languages
- [ ] Integration with court filing systems
- [ ] Advanced document templates
- [ ] Multi-jurisdiction support
- [ ] Document version control
- [ ] Legal precedent database integration

---

**Made with ❤️ for the Indian legal community**

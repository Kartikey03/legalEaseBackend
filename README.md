# LegalEaseAI: AI-Powered Legal Documentation Assistant

## Overview

LegalEaseAI is designed to simplify the process of creating legal documents for individuals and small businesses in India. The solution leverages AI to draft legal documents in plain language, making legal terminology accessible and easy to understand. This project consists of a Python backend, utilizes the BARD API for AI functionality, and features a frontend built with HTML, CSS, and JavaScript. The backend framework is Flask.

## Features

- **User-friendly Interface**: Easy-to-use interface for inputting relevant information such as case type, Petitioner Name, Respondent Name, Case Details and other necessary details.
- **AI-powered Document Generation**: Automatically drafts legal document in plain language and using easy-to-understand terms.
- **Customization**: Allows users to customize legal documents based on their specific needs.
- **Integration with Legal Resources**: Ensures the accuracy and completeness of the legal documents by integrating with existing legal resources.


## Requirements

- Python 3.7 or higher
- Flask
- BARD API credentials
- HTML/CSS/JS

## Setup Instructions
### 1. Clone the Repository

```
git clone https://github.com/kartikey03/LegalEase.git
cd legalEaseBackend
```

### 2. Set Up a Virtual Environment
It's a good practice to use a virtual environment to manage dependencies. You can set up a virtual environment using venv or virtualenv.

```bash
python3 -m venv legal_venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Backend Dependencies
Navigate to the backend directory and install the required Python packages.

```
pip install -r requirements.txt
```

### 4. Set Up BARD API
Ensure you have your BARD API credentials. You can set up environment variables to store these credentials securely.

```
export BARD_API_KEY='your_bard_api_key'
```

For Windows:
```
set BARD_API_KEY=your_bard_api_key
```
### 5. Run the Flask Application
Start the Flask application by running the main.py file.

```
python main.py
```
The application will be available at your localhost.

### 6. Access the Application
Open a web browser and navigate to http://localhost:5000/ to access the AI-Powered Legal Documentation Assistant.

### 7. Usage
Input Information: Enter the necessary details for your legal document.
Generate Document: Click the "Generate Legal Document" button to create a draft in plain language.
Customize Document: Modify the draft as needed to suit your specific requirements.

### 8. Contributing
We welcome contributions to improve this project! Please follow these steps to contribute:

### 9. Fork the repository.
Create a new branch for your feature or bugfix.
Make your changes.
Submit a pull request with a detailed explanation of your changes.

Thank you for using the LegalEaseAI! We hope it makes legal documentation easier and more accessible for you.

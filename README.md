# ChatGroq Observability RAG

ChatGroq Observability RAG is a Retrieval-Augmented Generation (RAG) system that leverages Groq's AI models to provide insights from the **LangSmith Observability** documentation. This tool helps users quickly retrieve and understand key concepts related to AI observability and tracing.

## 🚀 Features
- **Retrieval-Augmented Generation (RAG):** Fetches and processes documentation for precise answers.
- **Powered by Groq AI:** Uses `ChatGroq` for intelligent responses.
- **FAISS Vector Store:** Efficient document embedding and retrieval.
- **Web Scraping:** Loads and cleans documentation from the LangSmith site.
- **Streamlit UI:** Interactive chatbot interface.

## 📌 Installation
Ensure you have Python 3.8+ installed. Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/Langsmith-Observability-ChatGroq/chatgroq-observability-rag.git
cd chatgroq-observability-rag

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🔑 Environment Variables
Create a `.env` file in the root directory and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## 🎯 Usage
To start the Streamlit app, run:

```bash
streamlit run app.py
```

Then, open the displayed URL in your browser to interact with the chatbot.

## 🔍 How It Works
1. Loads **LangSmith Observability** documentation from the web.
2. Cleans and splits text into chunks.
3. Embeds the chunks using **Hugging Face's all-MiniLM-L6-v2** model.
4. Stores embeddings in **FAISS** for fast retrieval.
5. Uses **ChatGroq** to generate responses based on the retrieved documents.

## 📚 Tech Stack
- **FastAPI** (Optional for Backend)
- **Streamlit** (Frontend UI)
- **Groq AI** (LLM Provider)
- **FAISS** (Vector Search)
- **BeautifulSoup** (Web Scraping)
- **LangChain** (RAG Pipeline)

## 💡 Future Improvements
- Integrate multiple document sources.
- Enhance UI/UX for a better chatbot experience.
- Optimize retrieval speed and accuracy.

## 🤝 Contributing
Pull requests are welcome! Feel free to fork the repository and submit your contributions.

## 📜 License
This project is licensed under the MIT License.

## 📬 Contact
For questions or feedback, reach out to [odunukanabdulhameed@gmail.com].


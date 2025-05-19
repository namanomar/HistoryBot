# 📜 HistoryBot

**HistoryBot** is a conversational AI chatbot built with [LangChain](https://www.langchain.com/), [Google Gemini](https://ai.google.dev/), [FAISS](https://github.com/facebookresearch/faiss), and [Streamlit](https://streamlit.io/) that allows users to ask historical questions and get accurate, source-backed answers from a local knowledge base.

---

##  Features

- 📚 RAG (Retrieval-Augmented Generation) pipeline using FAISS vector store.
- 💬 Chat-style interface with memory using `st.chat_input` and `st.chat_message`.
- ⚡ Powered by **Google Gemini 1.5 Flash** for fast, factual responses.
- 🔎 Shows only the **top 1 relevant source document** used in answering.
- ☁️ Environment-agnostic (works locally or on Streamlit Cloud).

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/namanomar/historybot.git
cd historybot
```

### 2.  Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root folder with the following:
```bash
GOOGLE_API_KEY=your_google_api_key
```

4. Run the App
```
streamlit run app.py
```

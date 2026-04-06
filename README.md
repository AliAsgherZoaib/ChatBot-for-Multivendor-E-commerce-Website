# 🛒 ChatBot for Multivendor E-Commerce Website

A RAG-powered product recommendation chatbot built for e-commerce platforms. It combines **semantic search** (FAISS + Sentence Transformers) with a **Large Language Model** (Gemini 2.5 Flash) to deliver intelligent, context-aware product recommendations based on real customer reviews.

---

## 🚀 How It Works

The system is built in two phases:

**Phase 1 — Semantic Indexing**
1. Loads the [Amazon US Electronics Reviews dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) (TSV format)
2. Cleans the data — drops short reviews, keeps the single most helpful review per product
3. Encodes product titles + reviews using `all-MiniLM-L6-v2` (via Sentence Transformers)
4. Builds a FAISS vector index for fast cosine similarity search
5. Saves the index and product DataFrame to Google Drive

**Phase 2 — RAG + LLM Response**
1. Loads the saved FAISS index and product metadata
2. On user query, retrieves the top-k most semantically similar products
3. Optionally filters by star rating extracted from the query (e.g. "4 star headphones")
4. Constructs a structured prompt from retrieved products
5. Sends the prompt to **Gemini 2.5 Flash** to generate a human-friendly recommendation

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Search | `faiss-cpu` |
| LLM | Google Gemini 2.5 Flash (`google-genai`) |
| Data Processing | `pandas` |
| Runtime | Google Colab (T4 GPU recommended) |
| Storage | Google Drive |

---

## 📦 Installation

```bash
pip install sentence-transformers faiss-cpu pandas tqdm google-genai
```

---

## 🗂️ Dataset

This project uses the **Amazon Customer Reviews Dataset (Electronics)** available on Kaggle:

- File: `amazon_reviews_us_Electronics_v1_00.tsv`
- Upload it to your Google Drive and update the `filepath` variable in the notebook accordingly

---

## ⚙️ Setup & Usage

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Run Phase 1 — Build the Index
Open and run all cells in **Phase 1** to load, clean, embed, and index the data. The FAISS index and product DataFrame will be saved to your Drive.

### 3. Run Phase 2 — Start Querying
Load the saved artifacts and query the chatbot:

```python
query = "suggest a smart watch with good battery life and bluetooth calling feature"
print(recommend_with_llm(query, top_k=5))
```

### 4. Add Your Gemini API Key
Replace the placeholder in the notebook:
```python
client = genai.Client(api_key="YOUR_GEMINI_API_KEY_HERE")
```
Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 💡 Features

- **Semantic search** — finds relevant products even when exact keywords don't match
- **Review-based reasoning** — LLM uses real customer reviews to justify recommendations
- **Star rating filtering** — parses natural language like *"4 star headphones"* and filters accordingly
- **Price query handling** — gracefully informs users when price filtering isn't available
- **Batch embedding** — efficiently encodes large datasets in batches using GPU acceleration

---

## 📁 Project Structure

```
ChatBot_for_Multivendor_E_commerce_Website.ipynb
├── Phase 1: Data Loading & Cleaning
├── Phase 1: Embedding & FAISS Indexing
└── Phase 2: RAG Pipeline + Gemini LLM
```

---

## ⚠️ Notes

- The notebook is optimized for **Google Colab with T4 GPU**
- By default, only **500k rows** (~5 chunks of 100k) are loaded to keep runtime manageable — adjust the `chunks` limit as needed
- Make sure your FAISS index and DataFrame are aligned (same number of rows) before running Phase 2

---

## 📄 License

This project is open-source. Feel free to use, modify, and build on it.

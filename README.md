# 👗 Fashion Visual Search — Streamlit App

An AI-powered fashion visual search and recommendation tool built with **Streamlit**.  
This app uses OpenAI’s **CLIP** model and **FAISS** for image similarity search, helping users discover similar fashion items using image URLs.

---
## 📁 Project Structure

├── app/ # Core app modules
├── notebooks/ # Jupyter Notebooks for experimentation
├── requirements.txt # Python dependencies
├── streamlit_app.py # Entry point to run the Streamlit app
├── test_links.txt # Sample image URLs for testing
└── README.md # This file
---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Chiragjjjks/fashion-visual-search-Streamlit.git
cd fashion-visual-search-Streamlit
```

### 2️⃣ Set Up a Virtual Environment
✅ Recommended Python version: 3.11.2
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3️⃣ Install Dependencies
Step 1: Install from requirements.txt
```bash
pip install -r requirements.txt
```

Step 2: Install FAISS (separately for compatibility)
```bash
pip install faiss-cpu
```

▶️ Run the App
Launch the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

This will open the app in your default browser at:
👉 http://localhost:8501

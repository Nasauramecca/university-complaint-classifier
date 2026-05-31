import streamlit as st
import joblib
import string
import nltk
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import os
import streamlit as st
import nltk

# --- TRICK KHUSUS DEPLOYMENT NLTK CLOUD ---
# 1. Tentukan path folder nltk_data di server Streamlit
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")

# 2. Bikin foldernya kalau belum ada di server
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# 3. Masukkan path tersebut ke dalam list pencarian NLTK
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# 4. Paksa download resource satu per satu langsung ke folder tersebut
REQUIRED_RESOURCES = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
for resource in REQUIRED_RESOURCES:
    try:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Gagal download NLTK resource {resource}: {e}")
# ------------------------------------------

lemmatizer = WordNetLemmatizer()

def get_label(tag):
    if tag.startswith('j'):
        return 'a'
    elif tag.startswith('r') or tag.startswith('v') or tag.startswith('n'):
        return tag[0]
    else:
        return 'a'

def lemmatizing(word_list):
    lemma_list = []
    tagged = pos_tag(word_list)
    for word, tag in tagged:
        label = get_label(tag.lower())
        if label:
            result = lemmatizer.lemmatize(word, label)
            lemma_list.append(result)
        else:
            result = lemmatizer.lemmatize(word)
            lemma_list.append(result)
    return lemma_list

def preprocessing(sentence):
    eng_stopwords = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    sentence = sentence.lower()
    word_list = word_tokenize(sentence)
    filtered = [
        token for token in word_list
        if token not in eng_stopwords
        and token not in punctuations
        and token.isalpha()
    ]
    return lemmatizing(filtered)

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Naive Bayes": "nb_model.pkl",
        "Logistic Regression": "lr_model.pkl",
        "Linear SVC": "svc_model.pkl",
        "Random Forest": "rf_model.pkl"
    }
    for name, fname in model_files.items():
        if os.path.exists(fname):
            models[name] = joblib.load(fname)
    return models

DATA_FILE = "complaints_history.json"
def load_history():
    if os.path.exists(DATA_FILE):
        import json
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    import json
    with open(DATA_FILE, 'w') as f:
        json.dump(history, f)

if 'complaints' not in st.session_state:
    st.session_state.complaints = load_history()

# --- PAGE CONFIG & CUSTOM INTERACTIVE CSS (VS CODE THEME) ---
st.set_page_config(page_title="Complaint Classifier", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
    
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .code-font {
        font-family: 'Fira Code', monospace !important;
        font-weight: 500 !important;
    }
    
    h1 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        border-bottom: 1px solid #333333;
        padding-bottom: 10px;
        margin-bottom: 20px !important;
    }
    
    h3 {
        color: #85c5ec !important;
        font-size: 1.1rem !important;
        margin-bottom: 15px !important;
    }
    
    /* Input Form States (Interactive Focus) */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="textarea"] > div {
        background-color: #252526 !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 0px !important;
    }
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="textarea"]:focus-within > div {
        border-color: #007acc !important;
        box-shadow: 0 0 0 1px #007acc !important;
    }
    
    textarea {
        color: #d4d4d4 !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
    }
    
    /* Interactive Button States */
    .stButton > button {
        background-color: #0e639c !important;
        color: #ffffff !important;
        border: 1px solid #1177bb !important;
        border-radius: 0px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: background-color 0.1s ease !important;
    }
    .stButton > button:hover {
        background-color: #1177bb !important;
        border-color: #1488d8 !important;
    }
    
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: #2d2d2d !important;
        border: 1px solid #3c3c3c !important;
        color: #cccccc !important;
    }
    
    /* Fixed Terminal Scrollbar Box */
    .scroll-box {
        max-height: 380px !important;
        overflow-y: scroll !important;
        display: block;
        border: 1px solid #2d2d2d;
        background-color: #1a1a1a;
        padding: 10px;
    }
    
    .terminal-card {
        background-color: #252526;
        border-left: 4px solid #007acc;
        padding: 10px;
        margin-bottom: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
    }
    
    .status-tag {
        color: #4ec9b0;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    div[data-testid="stMetric"] {
        background-color: #252526 !important;
        border: 1px solid #2d2d2d !important;
        padding: 12px 16px !important;
        border-radius: 0px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Fira Code', monospace !important;
        color: #85c5ec !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Fira Code', monospace !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Header ---
st.markdown("<h1>University Complaint Classifier</h1>", unsafe_allow_html=True)

# --- Left-aligned Model Dropdown ---
col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    models = load_models()
    available_models = list(models.keys())
    if not available_models:
        st.error("No models found! Please save models from notebook first.")
        st.stop()
    selected_model = st.selectbox("Select Model Component", available_models)

st.write("##")

# --- Columns Architecture ---
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("### 📝 Input Complaint Stream", unsafe_allow_html=True)
    complaint = st.text_area("", height=150, placeholder="Type your complaint here...")
    
    st.write("")
    if st.button("Classify", type="primary", use_container_width=True):
        if complaint.strip():
            model = models.get(selected_model)
            if model:
                with st.spinner("Processing..."):
                    try:
                        pred = model.predict([complaint])[0]
                        st.session_state.complaints.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "complaint": complaint,
                            "genre": pred,
                            "model_used": selected_model
                        })
                        save_history(st.session_state.complaints)
                        
                        st.markdown(f"""
                        <div style="background-color: #233521; border: 1px solid #4ec9b0; padding: 12px; margin-top: 15px; font-family: 'Fira Code', monospace; font-size: 0.9rem; color: #4ec9b0;">
                            >> SUCCESS: Kategori terdeteksi -> [{pred}]
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Model not available")
        else:
            st.warning("Please enter a complaint")

with col_right:
    st.markdown("### 📋 Global Records (All Engines)", unsafe_allow_html=True)
    
    # 11 Daftar Kategori Lengkap Versi Panjang Sesuai Request Terbaru Lu
    GLOBAL_GENRES = [
        "Academic Support and Resources", 
        "Athletics and sports", 
        "Online learnings", 
        "Food and Cantines", 
        "Financial Support",
        "Health and Well-being Support",
        "Housing and Transportation",
        "International student experiences",
        "Internship Opportunities",
        "Student Affairs",
        "Student Activities"
    ]
    
    # Dropdown Filter dengan Nama Panjang
    selected_genre = st.selectbox("Filter by genre:", ["All"] + GLOBAL_GENRES)
    st.write("")
    
    if st.session_state.complaints:
        if selected_genre == "All":
            display_data = st.session_state.complaints[-10:]
        else:
            display_data = [c for c in st.session_state.complaints if c['genre'] == selected_genre][-10:]
        
        if display_data:
            # --- SCROLL BOX CONTAINER ---
            st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
            
            for item in reversed(display_data):
                st.markdown(f"""
                <div class="terminal-card">
                    <div class="status-tag">class: {item['genre']}</div>
                    <div style="color: #d4d4d4; margin-bottom: 6px; line-height: 1.4;">"{item['complaint']}"</div>
                    <div style="color: #6a9955; font-size: 0.75rem;">// {item['timestamp']} | engine: {item['model_used']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #6a9955; font-family: monospace;'>// no data available for this class</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #6a9955; font-family: monospace;'>// no predictions yet</div>", unsafe_allow_html=True)


# --- Analytics Dashboard Section ---
st.write("##")
st.markdown("<h2 style='font-size: 1.3rem !important; color:#ffffff; border-bottom: 1px solid #333333; padding-bottom: 6px;'>System Analysis Panel</h2>", unsafe_allow_html=True)

if st.session_state.complaints:
    df = pd.DataFrame(st.session_state.complaints)
    
    # METRICS ASLI LU
    a, b, c = st.columns(3)
    with a:
        st.metric("Total Complaints", len(df))
    with b:
        st.metric("Unique Categories", df['genre'].nunique())
    with c:
        most_common = df['genre'].mode().iloc[0] if not df.empty else "-"
        st.metric("Most Common", most_common)
    
    st.write("##")
    
    # DIAGRAM LINGKARAN (PIE CHART) DENGAN WARNA SESUAI GENRE
    genre_counts = df['genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    color_map = {
        "Academic Support and Resources": "#0e639c", 
        "Athletics and sports": "#34d399", 
        "Online learnings": "#c586c0", 
        "Food and Cantines": "#4ec9b0",
        "Financial Support": "#ce9178",
        "Health and Well-being Support": "#60a5fa",
        "Housing and Transportation": "#fbbf24",
        "International student experiences": "#fb7185",
        "Internship Opportunities": "#a78bfa",
        "Student Affairs": "#22d3ee",
        "Student Activities": "#f87171"
    }
    
    fig = px.pie(
        genre_counts, 
        names='Genre', 
        values='Count', 
        title="Complaints by Genre",
        template="plotly_dark",
        color='Genre',
        color_discrete_map=color_map
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#1e1e1e', width=2))
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(family="Fira Code, monospace", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # DELETE ACTION ASLI LU
    if st.button("Delete All Data"):
        st.session_state.complaints = []
        save_history([])
        st.rerun()
else:
    st.markdown("<div style='color: #6a9955; font-family: monospace; padding: 10px;'>// no data yet. start classifying!</div>", unsafe_allow_html=True)
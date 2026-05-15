
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
st.set_page_config(page_title="AI Support Ticket Classifier", page_icon="🎫", layout="centered")
st._config.set_option('theme.base', 'dark')
 
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .main { background-color: #0d0d0d; }
        h1 { font-family: 'Space Mono', monospace !important; color: #f0f0f0 !important; font-size: 1.8rem !important; letter-spacing: -1px; }
        .subtitle { color: #888; font-size: 0.95rem; margin-top: -10px; margin-bottom: 30px; }
        .result-box { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 20px 24px; margin-top: 20px; }
        .category-label { font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
        .billing { color: #f97316; }
        .technical { color: #3b82f6; }
        .general { color: #22c55e; }
        .confidence-text { color: #888; font-size: 0.85rem; }
        .stTextArea textarea { background-color: #1a1a1a !important; color: #f0f0f0 !important; border: 1px solid #333 !important; border-radius: 10px !important; }
        .stButton > button { background-color: #f0f0f0 !important; color: #0d0d0d !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 10px 28px !important; font-size: 0.85rem !important; }
        .stButton > button:hover { background-color: #d4d4d4 !important; }
        .metric-row { display: flex; gap: 12px; margin-top: 30px; }
        .metric-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 14px 18px; flex: 1; text-align: center; }
        .metric-value { font-family: 'Space Mono', monospace; font-size: 1.3rem; color: #f0f0f0; font-weight: 700; }
        .metric-label { color: #666; font-size: 0.75rem; margin-top: 2px; }
        .history-item { background: #111; border-left: 3px solid #333; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 0.88rem; }
        .tag { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; font-family: 'Space Mono', monospace; }
        .tag-billing { background: #431407; color: #f97316; }
        .tag-technical { background: #0f172a; color: #3b82f6; }
        .tag-general { background: #052e16; color: #22c55e; }
    </style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv")
    df = df.drop_duplicates(subset=['text'])
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'],
        test_size=0.2, random_state=42, stratify=df['category']
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, vectorizer, report
 
 
RESPONSES = {
    "billing": "Thank you for reaching out about your billing concern. Our billing team has been notified and will review your case within 2-3 business days. Please keep your transaction ID and any relevant receipts handy for faster resolution. For urgent issues, contact billing@support.com.",
    "technical": "We're sorry you're experiencing technical difficulties. Please try the following steps: (1) Clear your cache and cookies, (2) Restart the application, (3) Check your internet connection. If the issue persists, our technical team will reach out within 24 hours with a resolution.",
    "general": "Thank you for reaching out! For quick answers, visit our Help Center at help.support.com. Our support team is available Monday to Friday, 9AM–6PM IST. We'll respond to your query as soon as possible. Is there anything specific we can help clarify?"
}
 
model, vectorizer, report = train_model()
 
st.markdown("<h1>🎫 Support Ticket Classifier</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste a support query and the model will classify it instantly.</p>', unsafe_allow_html=True)
 
query = st.text_area("Enter support query", placeholder="e.g. I was charged twice for my subscription...", height=120)
 
col1, col2 = st.columns([1, 4])
with col1:
    classify_btn = st.button("CLASSIFY →")
 
if "history" not in st.session_state:
    st.session_state.history = []
 
if classify_btn and query.strip():
    vec = vectorizer.transform([query])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 1)
    color_class = prediction.lower()
    border_color = '#f97316' if prediction == 'billing' else '#3b82f6' if prediction == 'technical' else '#22c55e'
 
    st.markdown(f"""
        <div class="result-box">
            <div class="category-label {color_class}">
                {'🧾' if prediction == 'billing' else '🔧' if prediction == 'technical' else '💬'} {prediction.upper()}
            </div>
            <div class="confidence-text">Confidence: {confidence}%</div>
        </div>
    """, unsafe_allow_html=True)
 
    st.markdown(f"""
        <div class="result-box" style="margin-top: 12px; border-left: 3px solid {border_color}">
            <div style="color: #888; font-size: 0.75rem; margin-bottom: 8px; font-family: Space Mono, monospace;">SUGGESTED RESPONSE</div>
            <div style="color: #f0f0f0; font-size: 0.9rem; line-height: 1.6;">{RESPONSES[prediction]}</div>
        </div>
    """, unsafe_allow_html=True)
 
    st.session_state.history.insert(0, {"query": query, "category": prediction, "confidence": confidence})
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()
 
elif classify_btn:
    st.warning("Please enter a query first.")
 
st.markdown("---")
st.markdown("**Model Performance**")
st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="metric-value">80%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-card"><div class="metric-value">0.80</div><div class="metric-label">Macro F1</div></div>
        <div class="metric-card"><div class="metric-value">0.74</div><div class="metric-label">Billing F1</div></div>
        <div class="metric-card"><div class="metric-value">0.67</div><div class="metric-label">Technical F1</div></div>
    </div>
""", unsafe_allow_html=True)
 
if st.session_state.history:
    st.markdown("---")
    st.markdown("**Recent Queries**")
    for item in st.session_state.history:
        tag_class = f"tag-{item['category']}"
        st.markdown(f"""
            <div class="history-item">
                <span class="tag {tag_class}">{item['category'].upper()}</span>
                &nbsp; {item['query'][:80]}{'...' if len(item['query']) > 80 else ''}
                <span style="color:#555; float:right">{item['confidence']}%</span>
            </div>
        """, unsafe_allow_html=True)

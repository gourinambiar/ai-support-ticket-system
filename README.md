# 🎫 AI Support Ticket Classifier

An AI-powered support ticket classification system that automatically categorizes customer queries into **Billing**, **Technical**, and **General** categories using TF-IDF and Random Forest.

## 🚀 Live Demo
👉 [Click here to try it](https://ai-support-ticket-system-asytxrqlce5afvynixdrxh.streamlit.app/)

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| Macro F1 | 0.80 |
| Billing F1 | 0.74 |
| Technical F1 | 0.67 |

## 🛠️ Tech Stack
- Python
- Scikit-learn (TF-IDF + Random Forest)
- Streamlit
- Pandas, NumPy

## 📁 Project Structure
ai-support-bot/
├── app.py        # Streamlit frontend + backend
├── data.csv      # Training dataset
└── requirements.txt
## 🔧 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
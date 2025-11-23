# 👕 ClosetAI — Intelligent Weather-Adaptive Outfit Recommendation System

ClosetAI is an end-to-end AI system that recommends outfits based on the user’s closet, weather conditions, the season, and natural-language prompts (e.g., “going to a party,” “business meeting,” “gym”).  
The system integrates:

- Semantic embeddings (SentenceTransformer)
- NLP intent parsing (spaCy)
- Weather-aware scoring logic
- Differential privacy on statistics
- Structured data validation
- A full Gradio interface for real-time interaction
- Modular architecture with clear trustworthiness, HCI, and lifecycle components

This project was developed for **EGN 6216 Artificial Inteligence Sytems** and follows NIST AI RMF, privacy, and HCI guidelines.

---

## 📁 Repository Contents

### `src/`
Core system code:

- `main.py` — system entry point
- `model_pipeline.py` — preprocessing, categorization, scoring, and outfit generation
- `data_validation.py` — schema checks, imbalance analysis, differential privacy, risk logs
- `ui.py` — Gradio user interface
- `utils.py` — shared helper functions (optional)

### `deployment/`
Containerization assets:

- `Dockerfile` (planned or optional)
- `environment.yml` or `requirements.txt`
- Run instructions for launching in a cloud or container environment

### `monitoring/`
Performance monitoring configuration:

- Scripts for collecting latency, scoring metrics
- Prometheus/Grafana config (optional)
- Example exported metrics (CSV or JSON)

### `documentation/`
All written reports and templates:

- AI System Project Proposal  
- System Architecture Report  
- Performance Metrics Report  
- Risk Analysis Notes  
- HCI Wireframes / Screenshots  

### `videos/`
Demo screencasts showing:

- Full system workflow  
- Gradio UI in action  
- Data validation logs  
- Model outputs under multiple scenarios  

---

## ▶️ **System Entry Point**

The primary script is: src/main.py


### **Running Locally**

From project root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.main
```

Then visit:
```bash
http://127.0.0.1:7860

ClosetAI/
│── src/
│   ├── main.py
│   ├── ui.py
│   ├── model_pipeline.py
│   ├── data_validation.py
│   └── utils.py
│
│── documentation/
│── deployment/
│── monitoring/
│── videos/
│── requirements.txt
│── README.md


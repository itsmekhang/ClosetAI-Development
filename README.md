# ClosetAI — Intelligent Weather-Adaptive Outfit Recommendation System

ClosetAI is an end-to-end AI system that recommends outfits based on the user’s closet, weather conditions, the season, and natural-language prompts (e.g., “going to a party,” “business meeting,” “gym”).  
The system integrates:

- Semantic embeddings (SentenceTransformer)
- NLP intent parsing (spaCy)
- Weather-aware scoring logic
- Differential privacy
- Structured data validation
- A full Gradio interface for real-time interaction
- Modular architecture with clear trustworthiness, HCI, and lifecycle components

This project was developed for **EGN 6216 Artificial Inteligence Sytems** and follows NIST AI RMF, privacy, and HCI guidelines.

---

## 2. Repository Contents

### `src/`
Core system code:

- `main.py` — system entry point
- `model_pipeline.py` — preprocessing, categorization, scoring, and outfit generation
- `data_validation.py` — schema checks, imbalance analysis, differential privacy, risk logs
- `ui.py` — Gradio user interface
- 
## **3. System Entry Point**

The primary script is: 
```bash
src/main.py
```
### 4. Video Demonstration
[Watch System Demo](videos/demo_video.mp4)
### 5. Deployment Strategy
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
```
Containerization assets:

- `Dockerfile` (planned or optional)
- `environment.yml` or `requirements.txt`
- Run instructions for launching in a cloud or container environment

### 6. Monitoring and Metrics
Performance monitoring configuration:

- Scripts for collecting latency, scoring metrics
- Example exported metrics (CSV or JSON)

### 7. Documentation
All written reports and templates:

- AI System Project Proposal  
- [Project Report](documentation/project%20report.pdf)



Repository structure:
```bash
ClosetAI/
│── src/
│   ├── main.py
│   ├── ui.py
│   ├── model_pipeline.py
│   ├── data_validation.py
│   ├──data/
|      ├── closet.csv
|      ├── imbalance_report.csv
|      ├── metadata.json
|      ├── occasion_distribution.csv
|      ├── risk.csv
|      ├── season_distribution.csv
|
│── documentation/
|   ├── README.md
|   ├── project report.pdf
|
│── deployment/
│── monitoring/
|   ├──metrics_log.csv
|
│── videos/
│── requirements.txt
│── README.md
```

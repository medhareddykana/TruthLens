# 🚀 TruthLens Quantum

TruthLens Quantum is an AI-powered misinformation detection, education, and response platform, built for a hackathon. It provides a real-time "TruthScore" for text and images, visualizes misinformation trends, and empowers users with an AI Coach and counter-narrative generator.

## ✨ Features

* **Multi-Modal Analysis:** Analyzes both text and images to detect potential misinformation.
* **AI TruthScore:** Assigns a credibility score based on an analysis of factual claims.
* **AI Coach:** Provides personalized media literacy tips to help users identify misinformation techniques.
* **Counter-Narratives:** Instantly generates fact-based responses to low-scoring content.
* **Live Dashboard:** Visualizes misinformation "hot topics" and the propagation of sources through an interactive graph.
* **Recent Analyses Feed:** Shows a live feed of the most recently analyzed content.

## 🛠️ Tech Stack

* **Frontend:** HTML, Tailwind CSS, D3.js
* **Backend:** Python, FastAPI
* **AI:** Google Gemini
* **Database:** Simple JSON file for logging (`analysis_log.json`)

## ⚙️ Running Locally

1.  Clone the repository.
2.  Set up a Python virtual environment and install dependencies from `requirements.txt`.
3.  Create a `.env` file with your `GEMINI_API_KEY`.
4.  Run the backend server with `python -m uvicorn main:app --reload`.
5.  Open the `index.html` file in your browser.

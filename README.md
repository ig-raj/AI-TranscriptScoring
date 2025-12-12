# AI Transcript Scoring Tool

A lightweight FastAPI + NLP based transcript scoring system for student self-introductions.

## Features
- Score out of 100
- Per-criterion breakdown
- Word count, sentence count, WPM
- Clarity, grammar, sentiment evaluation
- Semantic similarity for understanding meaning

## Tech Stack
- FastAPI
- SentenceTransformer (all-MiniLM-L6-v2)
- HTML/CSS/JS Frontend
- Docker (Hugging Face Deployment)

## Run Locally
uvicorn app.main:app --reload

## Frontend
Open /frontend/index.html in browser.

## Deploy
Push repo to Hugging Face Space using Dockerfile.
Ensure app listens on port 7860.

## Limitations
- Text-only (no audio)
- Heuristic grammar checks
- CPU embedding model

## Future Enhancements
- Real-time speech scoring
- Teacher dashboard analytics
- Strong grammar engine integration

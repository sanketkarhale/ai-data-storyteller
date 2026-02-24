# AI-Powered Data Storyteller

A Streamlit web app that:
- Uploads a CSV dataset
- Shows preview + summary statistics
- Creates basic visualizations (histogram, bar chart, scatter)
- Optionally uses OpenAI API to generate a human-readable data story

## Project Structure

```text
ai-data-storyteller/
├── app.py
├── requirements.txt
├── .env               # contains OPENAI_API_KEY (not committed)
└── data/
    └── Superstore.csv # sample dataset


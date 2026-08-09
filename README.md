# CEUIL AI News-to-MCQ Pipeline

A Python automation project that collects news articles and generates multiple-choice questions from them using a local language model.

## Workflow

1. `news_scraper.py` collects articles into the `articles/` directory.
2. `mcq_generator.py` processes collected material into quiz questions.
3. GitHub Actions can run the scraper and generator on schedules or manually.

## Requirements

- Python 3.10+
- Python dependencies listed in `.github/workflows/requirements.txt`
- Local model/runtime support for the configured `llama-cpp-python` workflow

## Run locally

```bash
python -m pip install -r .github/workflows/requirements.txt
python news_scraper.py
python mcq_generator.py
```

## Notes

Model files and generated quizzes can be large. Keep them out of version control when possible, and use a small sample dataset for reproducible demonstrations.

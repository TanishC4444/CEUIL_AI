# CEUIL AI News-to-MCQ Pipeline

A Python automation pipeline that collects news articles and generates multiple-choice questions from them using a local language model.

## Overview

The project connects article collection with automated educational question generation. GitHub Actions can run the pipeline on a schedule or manually, while local model inference keeps generation under the project's control.

## Features

- News article collection
- Automated MCQ generation
- Local language-model inference
- Scheduled and manual GitHub Actions workflows
- Separate scraper and generator stages

## Prerequisites

- Python 3.10+
- Dependencies in `.github/workflows/requirements.txt`
- A supported local `llama-cpp-python` runtime and model

## Installation

```bash
git clone https://github.com/TanishC4444/CEUIL_AI.git
cd CEUIL_AI
python -m pip install -r .github/workflows/requirements.txt
```

## Quick Start

```bash
python news_scraper.py
python mcq_generator.py
```

The scraper populates `articles/`; the generator processes the collected material into quiz questions.

## Automation

GitHub Actions supports scheduled and manual executions. Keep model files, credentials, and large generated datasets out of source control.

## Project Structure

```text
CEUIL_AI/
├── news_scraper.py
├── mcq_generator.py
├── articles/
└── .github/workflows/
```

## Status

Active development project.

## License

No separate license is currently specified in the repository.

## Support

Use GitHub Issues for bugs and questions.

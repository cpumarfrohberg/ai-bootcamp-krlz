FROM python:3.11-slim

ENV TMPDIR=/tmp

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN TMPDIR=/tmp uv pip install --system -e .

COPY config/ ./config/
COPY wikiagent/ ./wikiagent/
COPY streamlit_app.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

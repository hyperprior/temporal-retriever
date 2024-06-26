FROM python:3.12.2-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . $HOME/.cargo/env && \
    uv pip install -r requirements.txt --system --compile && \
    apt-get remove -y curl gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY temporal_retriever temporal_retriever

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "temporal_retriever.app:app", "--host", "0.0.0.0", "--port", "8000"]

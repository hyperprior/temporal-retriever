FROM python:3.12.2-slim

COPY pyproject.toml poetry.lock .

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev --no-interaction --no-cache && \
    rm -rf /root/.cache

COPY temporal_retriever /temporal_retriever

RUN useradd -m appuser
USER appuser

EXPOSE 8000

ENTRYPOINT ["uvicorn", "temporal_retriever.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11

COPY pyproject.toml poetry.lock .

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev --no-interaction --no-cache && \
    rm -rf /root/.cache

COPY temporal_retriever .

EXPOSE 8000

CMD ["ls"]


# ENTRYPOINT ["uvicorn", "temporal_retriever.app:app", "--host", "0.0.0.0", "--port", "8000"]

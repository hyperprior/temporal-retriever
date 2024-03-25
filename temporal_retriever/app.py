from fastapi import FastAPI, status

app: FastAPI = FastAPI()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return


@app.post("/analyze")
async def analyze_datasets(request: None) -> None:
    pass

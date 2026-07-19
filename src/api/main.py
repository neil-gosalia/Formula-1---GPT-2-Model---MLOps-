from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import GenerateRequest, GenerateResponse
from model import load_model, generate_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="F1 GPT-2 API",
    description="Fine tuned GPT-2 Model for F1 text generation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def health_check():
    return {"status": "ok", "model": "f1 text generation"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    input_text = request.prompt
    top_p = request.top_p
    temperature = request.temperature
    max_length = request.max_length
    try:
        output = generate_text(
            prompt=input_text,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length
        )
        return GenerateResponse(response=output, prompt=input_text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

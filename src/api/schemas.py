from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    top_p: float = 0.95
    temperature: float = 0.80
    max_length: int = 200


class GenerateResponse(BaseModel):
    response: str
    prompt: str

import time
from typing import List
import os
from yandex_cloud_ml_sdk import YCloudML

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger

# Initialize
app = FastAPI()
sdk = YCloudML(folder_id=os.getenv("CATALOG_ID"), auth=os.getenv("API_KEY"))
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode('utf-8')}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode('utf-8')}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        # Здесь будет вызов вашей модели
        messages = [
            {
                "role": "system",
                "text": "Ответь на следующий вопрос, указав номер правильного варианта ответа и краткое пояснение",
            },
            {
                "role": "user",
                "text": body.query,
            },
                ]
        result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages).alternatives[0].text

        answer = 1  # Замените на реальный вызов модели
        sources: List[HttpUrl] = [
            # HttpUrl("https://itmo.ru/ru/"),
            # HttpUrl("https://abit.itmo.ru/"),
        ]

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=result,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.llm_routes import router as llm_router
from api.routes.model_routes import router as model_router

app = FastAPI(title="Cricket Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm_router)
app.include_router(model_router)

@app.get("/")
def root():
    return {"status": "ok", "routes": ["/llm/predict", "/model/predict", "/model/filter"]}

# Run with: uvicorn main:app --reload

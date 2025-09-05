from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from api.utils.llm import generate_response

router = APIRouter(prefix="/llm", tags=["llm"])

class LlmExplainRequest(BaseModel):
    """Inputs for generating a natural-language explanation.

    The LLM summarizes why the predicted outcome is plausible given the
    inputs, referencing run rates, overs left, and wickets in hand.
    """
    # inputs and predicted result only
    total_runs: float
    wickets: float
    target: int
    balls_left: float
    predicted: int
    probability: float

@router.post("/predict")
def llm_predict(payload: LlmExplainRequest):
    """Return a concise model explanation using the configured OpenAI model."""
    try:
        system_prompt = (
            "You are an assistant that explains a cricket chase outcome prediction succinctly and clearly. "
            "Keep the explanation factual and refer to run rates, overs left, and wickets in hand."
        )
        user_prompt = (
            f"Inputs: total_runs={payload.total_runs}, wickets={payload.wickets}, target={payload.target}, "
            f"balls_left={payload.balls_left}. Model predicted won={payload.predicted} (p={payload.probability:.3f}).\n"
            "Explain in 2-4 sentences why this outcome is plausible given the inputs."
        )
        response = generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4.1",
            temperature=0.4,
            response_format=None,
        )
        return {"explanation": response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



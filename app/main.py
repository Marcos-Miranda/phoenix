import logging
from typing import Literal

from fastapi import FastAPI

from app import otel_config, phoenix_config, workflow_custom, workflow_llama
from app.config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

if settings.tracer_method == "otel":
    otel_config.set_tracer_provider(app)
else:
    phoenix_config.set_tracer_provider()


@app.get("/")
async def home() -> str:
    return "Welcome!"


@app.post("/chat/")
async def chat_endpoint(user_question: str, type: Literal["custom", "llama"] = "custom") -> str:
    wf_cls = workflow_custom.MyWorkflow if type == "custom" else workflow_llama.MyWorkflow
    workflow = wf_cls()
    response = await workflow.run(user_question=user_question)
    return response

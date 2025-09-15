import logging

from fastapi import FastAPI

from app import otel_config, phoenix_config
from app.config import settings
from app.workflow import DummyWorkflow, LLamaWorkflow

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
if settings.tracer_method == "otel":
    otel_config.set_tracer_provider()
else:
    phoenix_config.set_tracer_provider()


app = FastAPI()


@app.get("/")
async def home() -> str:
    return "Welcome!"


@app.post("/chat/")
async def chat_endpoint(user_message: str, type: str = "dummy") -> str:
    wf_cls = DummyWorkflow if type == "dummy" else LLamaWorkflow
    workflow = wf_cls()
    response = await workflow.run(user_message=user_message)
    return response

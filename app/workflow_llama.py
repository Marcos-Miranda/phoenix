from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from app.config import settings
from app.shared import ASSISTANT_SYSTEM_PROMPT, GUARDRAIL_FAILURE_MESSAGE, GuardrailOutput, example_traced_operation


def init_model():
    if not settings.model.startswith("gpt-"):
        return Ollama(model=settings.model, temperature=0, request_timeout=30)
    return OpenAI(
        model=settings.model,
        temperature=0,
        api_base=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=30,
        max_retries=1,
    )


model = init_model()
structured_model = model.as_structured_llm(GuardrailOutput)


class State(BaseModel):
    user_message: ChatMessage = ChatMessage(role=MessageRole.USER, content="")
    assistant_message: ChatMessage = ChatMessage(role=MessageRole.ASSISTANT, content="")
    guardrail_response: GuardrailOutput = GuardrailOutput(is_compliant=True, reasoning="")


class AssistantEvent(Event):
    pass


class GuardrailEvent(Event):
    pass


class MyWorkflow(Workflow):
    @step
    async def start(self, ctx: Context[State], ev: StartEvent) -> AssistantEvent:
        _ = await example_traced_operation()
        async with ctx.store.edit_state() as state:
            state.user_message.content = ev.user_question
        return AssistantEvent()

    @step
    async def assistant(self, ctx: Context[State], ev: AssistantEvent) -> GuardrailEvent:
        async with ctx.store.edit_state() as state:
            response = await model.achat(
                [ChatMessage(role=MessageRole.SYSTEM, content=ASSISTANT_SYSTEM_PROMPT), state.user_message]
            )
            state.assistant_message.content = response.message.content
        return GuardrailEvent()

    @step
    async def guardrail(self, ctx: Context[State], ev: GuardrailEvent) -> StopEvent:
        async with ctx.store.edit_state() as state:
            response = await structured_model.achat([state.user_message, state.assistant_message])
            state.guardrail_response = response.raw
            return StopEvent(
                result=state.assistant_message.content
                if state.guardrail_response.is_compliant
                else GUARDRAIL_FAILURE_MESSAGE
            )

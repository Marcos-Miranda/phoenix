from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# from opentelemetry import trace
from pydantic import BaseModel, Field

from app.config import settings, tracer


class GuardrailOutput(BaseModel):
    """Non-toxic output validation."""

    is_compliant: bool = Field(
        description="Indicates if the assistant's message is compliant with non-toxicity standards"
    )
    reasoning: str = Field(description="Reasoning behind the compliance decision")


openai_client = AsyncOpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    max_retries=1,
    timeout=30,
)


async def answer_question(user_message: ChatCompletionUserMessageParam) -> ChatCompletionAssistantMessageParam:
    system_message = ChatCompletionSystemMessageParam(
        role="system",
        content="You are a helpful assistant but sarcastic when the question is dumb.",
    )
    response = await openai_client.chat.completions.create(
        model=settings.model,
        messages=[system_message, user_message],
        temperature=0,
    )
    return response.choices[0].message


async def validate_response(
    user_message: ChatCompletionUserMessageParam, assistant_message: ChatCompletionAssistantMessageParam
) -> GuardrailOutput:
    response = await openai_client.chat.completions.parse(
        model=settings.model,
        messages=[user_message, assistant_message],
        response_format=GuardrailOutput,
        temperature=0,
    )
    parsed = response.choices[0].message.parsed
    if not parsed:
        raise ValueError("Failed to parse the guardrail response")
    return parsed


class State(BaseModel):
    user_message: ChatCompletionUserMessageParam = ChatCompletionUserMessageParam(role="user", content="")
    assistant_message: ChatCompletionAssistantMessageParam = ChatCompletionAssistantMessageParam(
        role="assistant", content=""
    )
    guardrail_response: GuardrailOutput = GuardrailOutput(is_compliant=True, reasoning="")


class AssistantEvent(Event):
    pass


class GuardrailEvent(Event):
    pass


class LLamaWorkflow(Workflow):
    @step
    async def start(self, ctx: Context[State], ev: StartEvent) -> AssistantEvent:
        async with ctx.store.edit_state() as state:
            state.user_message = ChatCompletionUserMessageParam(role="user", content=ev.user_message)
        return AssistantEvent()

    @step
    async def assistant(self, ctx: Context[State], ev: AssistantEvent) -> GuardrailEvent:
        async with ctx.store.edit_state() as state:
            state.assistant_message = await answer_question(state.user_message)
        return GuardrailEvent()

    @step
    async def guardrail(self, ctx: Context[State], ev: GuardrailEvent) -> StopEvent:
        async with ctx.store.edit_state() as state:
            state.guardrail_response = await validate_response(state.user_message, state.assistant_message)
            return StopEvent(
                result=state.assistant_message.content
                if state.guardrail_response.is_compliant
                else "I'm sorry, I cannot assist with that request because my response was flagged as inappropriate."
            )


class DummyWorkflow:
    def __init__(self) -> None:
        self.state = State()

    @tracer.chain
    async def start(self, user_message: str) -> None:
        self.state.user_message = ChatCompletionUserMessageParam(role="user", content=user_message)

    @tracer.chain
    async def assistant(self) -> None:
        self.state.assistant_message = await answer_question(self.state.user_message)

    @tracer.chain
    async def guardrail(self) -> str:
        self.state.guardrail_response = await validate_response(self.state.user_message, self.state.assistant_message)
        return (
            self.state.assistant_message.content
            if self.state.guardrail_response.is_compliant
            else "I'm sorry, I cannot assist with that request because my response was flagged as inappropriate."
        )

    @tracer.chain
    async def run(self, user_message: str) -> str:
        await self.start(user_message)
        await self.assistant()
        response = await self.guardrail()
        return response

from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, ConversationalGEval
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams

from chatbot import chat

turns=[]
history=[]
for user_msg in[
       "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        " jusWhat was the ETA yout mentioned?",   # tests memory retention
        "Can I upgrade to express shipping?",

]:
    reply,history,tools=chat(user_msg,history)
    turns.append(Turn(role="user",content=user_msg))
    turns.append(Turn(role="assistant",content=reply))

import os
from deepeval.models import LiteLLMModel

from deepeval.models import LiteLLMModel
class GroqJudge(LiteLLMModel):
    def generate_raw_response(self, prompt, top_logprobs=5, audio=None):
        raise AttributeError("logprobs not supported by groq gpt-oss")

    async def a_generate_raw_response(self, prompt, top_logprobs=5, audio=None):
        raise AttributeError("logprobs not supported by groq gpt-oss")

model = GroqJudge(model="groq/openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

Correctness=ConversationalGEval(
    name="Correctness",
    criteria=("Did the chatbot fully resolve the customer's issue? "
      "It should use tools when needed and provide accurate answers."
),
    model=model,
    threshold= 0.7,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
])

test_case=ConversationalTestCase(
    turns=turns

)
evaluate(test_cases=[test_case],metrics=[Correctness])
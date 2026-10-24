import os

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.evaluate import AsyncConfig
from deepeval.tracing import observe
from deepeval.models import LiteLLMModel

from agent_instrumented import support_agent as  _support_agent

model= LiteLLMModel(model="openrouter/nvidia/nemotron-3.5-super-vl:free",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0,
                    )

@observe(name="support_agent")
def support_agent(user_input: str) -> str:

    return _support_agent(user_input)

answer_relevancy= AnswerRelevancyMetric(threshold=0.7,model=model)
step_efficiency= StepEfficiencyMetric(threshold=0.5,model=model)
prompt_alignment = PromptAlignmentMetric(
    threshold=0.7,
    model=model,
    prompt_instructions=[
        "You are a friendly customer-support agent",
        "Use the available tools to answer order and refund questions",
        "Keep replies short and helpful",
    ],
)

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Where is my order ORD-1042?",
        ),
        Golden(
            input="What is the refund policy of electronics",
        ),
    ]
)


for golden in dataset.evals_iterator(metrics=[answer_relevancy,step_efficiency]):
    async_config = AsyncConfig(run_async=False)
    response = support_agent(golden.input)
    print(f"\nQuestion: {golden.input}")
    print(f"answer: {response}\n")





from aiohttp.hdrs import FROM
# THIS IS WHITE BOX TESTING

from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall
from deepeval.evaluate import AsyncConfig
from deepeval.tracing import observe, update_current_trace
from deepeval.contextvars import get_current_golden
from agent_instrumented import support_agent as base_agent



@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return base_agent(user_input)

# model ////
from deepeval.models import LiteLLMModel

LiteLLMModel(model="")
task_completion = TaskCompletionMetric(threshold=0.7, model=router)
tool_correctness = ToolCorrectnessMetric(threshold=0.7, model=router)


dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Where is my order ORD-1042?",
            expected_tools=[ToolCall(name="get_order_status")],
        ),
        Golden(
            input="What is the refund policy of electronics",
            expected_tools=[ToolCall(name="get_refund_policy")],
        ),
    ]
)

# AsyncConfig(run_async=False) using this 1 test will run on one time (if you use this you won't see any rat limits if your ai model is low level or you have not enough tokens!)
for golden in dataset.evals_iterator(
    metrics=[task_completion, tool_correctness],
    async_config=AsyncConfig(run_async=True),
):
    response = support_agent(golden.input)

    #responce of ai
    print("\n" + "=" * 50)
    print(f"❓ Question of user: {golden.input}")
    print(f"🤖 reply of ai: {response}")
    print("=" * 50 + "\n")
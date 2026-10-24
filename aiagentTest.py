from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from agent_instrumented import support_agent
from groq_evaluator import GroqEvaluator
from deepeval.metrics import ToolCorrectnessMetric
from cryptography.x509 import name

#user_input = "Where is my order ORD-1042?"
#actual_output = support_agent(user_input)

# test_case = LLMTestCase(
#     input=user_input,
#     actual_output=actual_output          # THIS IS BLACK BOX TESTING
# )
# print("\n" + "="*50)
# print(f"🤖 AGENT RESPONSE:\n{actual_output}")
# print("="*50 + "\n")

# evaluate(
#     test_cases=[test_case],
#     metrics=[
#         TaskCompletionMetric(
#             threshold=0.7,
#             model=GroqEvaluator()
#         )
#     ]
# )

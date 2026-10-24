
from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric, \
    AnswerRelevancyMetric
from deepeval.tracing import observe, update_current_trace

from newMetrics import model
from rag_agent import rag_support_agent as _rag_support_agent


@observe(name="rag_support_agent" )
def rag_support_agent(user_input: str) -> str:
    golden= get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _rag_support_agent(user_input)


contextualPrecision_metric= ContextualPrecisionMetric(
    threshold=0.7,
    model=""
)
conRecall_metric=ContextualRecallMetric(
    threshold=0.7,
    model=""
)
faithfulness_metric= FaithfulnessMetric(
    threshold=0.7,
    model=""
)
answerRel_metric=AnswerRelevancyMetric(
    threshold=0.7,
    model=""
)


dataset= EvaluationDataset(goldens=[Golden(
    input="What is the return policy for electronics?",
     expected_output=("Electronics can be returned within 15 days of delivery if unopened "
                     "and in original packaging. Refunds take 5–7 business days.")),
    Golden(input="How long does express shipping take and what does it cost?",
           expected_output=("Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."))])



for golden in dataset.evals_iterator(metrics=[contextualPrecision_metric,conRecall_metric,answerRel_metric,faithfulness_metric]):
    async_config = AsyncConfig(run_async=False)
    rag_support_agent(golden.input)

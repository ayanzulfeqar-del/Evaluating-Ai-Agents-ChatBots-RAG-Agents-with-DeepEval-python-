from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.test_case import ConversationalTestCase, Turn
from chatbot import chat



print(turns)
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

    model=LiteLLMModel(model="groq/openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

turn_relevancy= TurnRelevancyMetric(threshold=0.6,model=model)
# knowledge_retention=KnowledgeRetentionMetric(threshold=0.7,model=LiteLLMModel())
# conversation_comMetric=ConversationCompletenessMetric(threshold=0.7,model=model)


test_case=ConversationalTestCase(
    turns=turns

)
evaluate(test_cases=[test_case],metrics=[turn_relevancy])
from transformers import pipeline

# 1. Basic Pipeline Setup
qa_pipeline_default = pipeline("question-answering")

context_1 = """
Charles Babbage was an English mathematician, philosopher, inventor and mechanical engineer who originated the concept of a programmable computer.
"""
question_1 = "Who is known as the father of the computer?"

result_1 = qa_pipeline_default(question=question_1, context=context_1)
print("Basic Pipeline Result:\n", result_1)

# 2. Use a Custom Pretrained Model
qa_pipeline_custom = pipeline("question-answering", model="deepset/roberta-base-squad2")
result_2 = qa_pipeline_custom(question=question_1, context=context_1)
print("\nCustom Model Result:\n", result_2)

# 3. Test on Your Own Example
custom_context = """
The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Colombia, Peru, and other South American countries.
It is known for its biodiversity and is often referred to as the lungs of the planet.
"""

question_3a = "What is the largest tropical rainforest in the world?"
question_3b = "Why is the Amazon rainforest important?"

result_3a = qa_pipeline_default(question=question_3a, context=custom_context)
result_3b = qa_pipeline_default(question=question_3b, context=custom_context)

print("\nCustom Context Question 1:\n", result_3a)
print("\nCustom Context Question 2:\n", result_3b)

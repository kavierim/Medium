# Make sure you have run: pip install transformers torch datasets evaluate
from transformers import pipeline

# Load the Question Answering pipeline
qa_pipeline = pipeline("question-answering")

# Define the context (where the answer lies)
context = """
The Apollo 11 mission, launched on July 16, 1969, was the first mission to land humans on the Moon.
The commander was Neil Armstrong, the command module pilot was Michael Collins, and the lunar module pilot was Buzz Aldrin.
Armstrong became the first person to step onto the lunar surface on July 21, 1969, followed by Aldrin.
"""

# Define the question
question = "Who was the commander of the Apollo 11 mission?"

# Perform Question Answering
qa_result = qa_pipeline(question=question, context=context)

# Print the result
print(f"Context:\n{context}")
print(f"Question: {question}")
print("\nAnswer Found:")
if qa_result:
    print(f"- Answer: {qa_result['answer']}")
    print(f"  Score: {qa_result['score']:.4f}")
    print(f"  Start Index: {qa_result['start']}")
    print(f"  End Index: {qa_result['end']}")
else:
    print("Could not find an answer in the context.")

# Example 2
question_2 = "When was the mission launched?"
qa_result_2 = qa_pipeline(question=question_2, context=context)
print(f"\nQuestion: {question_2}")
print("\nAnswer Found:")
if qa_result_2:
    print(f"- Answer: {qa_result_2['answer']}")
    print(f"  Score: {qa_result_2['score']:.4f}")
else:
    print("Could not find an answer in the context.")


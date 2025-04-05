# Make sure you have run: pip install transformers torch datasets evaluate
from transformers import pipeline

# Load the NER pipeline
# 'grouped_entities=True' combines parts of the same entity (e.g., "New" and "York")
ner_pipeline = pipeline("ner", grouped_entities=True)

# Example text
text = "My name is Clara and I live in Berlin. I work for the European Space Agency."

# Perform NER
ner_results = ner_pipeline(text)

# Print the results nicely using standard Python
print(f"Text: \"{text}\"")
print("\nEntities Found:")
if ner_results:
    for entity in ner_results:
        print(f"- Entity: {entity['word']}")
        print(f"  Label: {entity['entity_group']}")
        print(f"  Score: {entity['score']:.4f}")
else:
    print("No entities found.")


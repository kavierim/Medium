# Make sure you have run: pip install transformers torch sentencepiece
# Note: sentencepiece is often required for multilingual tokenizers
from transformers import pipeline

# Load translation pipeline for English to French
translator_en_fr = pipeline("translation_en_to_fr")

# Example text
text_en = "Hello, how are you today?"

# Perform translation
translation_fr = translator_en_fr(text_en)

# Print the result (pipeline usually returns a list)
print(f"Original (en): \"{text_en}\"")
if translation_fr:
    print(f"Translation (fr): \"{translation_fr[0]['translation_text']}\"")
else:
    print("Translation failed.")

# --- Example 2: English to German ---
print("\n--- Another Language Pair ---")
try:
    # Load translation pipeline for English to German
    # This might download a different model automatically
    translator_en_de = pipeline("translation_en_to_de")
    text_en_2 = "Transformers are powerful models for natural language processing."
    translation_de = translator_en_de(text_en_2)

    print(f"Original (en): \"{text_en_2}\"")
    if translation_de:
        print(f"Translation (de): \"{translation_de[0]['translation_text']}\"")
    else:
        print("Translation failed.")

except Exception as e:
    print(f"Could not load or run en->de translation pipeline: {e}")
    print("Ensure you have an internet connection and necessary model files can be downloaded.")


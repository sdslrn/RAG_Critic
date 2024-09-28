import difflib
from fuzzywuzzy import fuzz

text1 = """
Squamous cell carcinoma of the lung may be classified according to the WHO histological classification system into 4 main types: papillary, clear cell, small cell, and basaloid.
"""
text2 = """
Common risk factors in the development of squamous cell carcinoma of the lung include smoking, family history of lung cancer, high levels of air pollution, radiation therapy to the chest, radon gas, asbestos, occupational exposure to chemical carcinogens, and previous lung disease.
"""
similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
print("difflib similarity: ", similarity)

similarity = fuzz.ratio(text1, text2)
print("fuzzywuzzy ratio similarity: ", similarity)

# 使用 partial_ratio
partial_similarity = fuzz.partial_ratio(text1, text2)

# 使用 token_set_ratio
token_set_similarity = fuzz.token_set_ratio(text1, text2)

print(f"Partial Ratio: {partial_similarity}")
print(f"Token Set Ratio: {token_set_similarity}")

token_set_similarity = fuzz.token_set_ratio(text2, text1)

print(f"Partial Ratio: {partial_similarity}")
print(f"Token Set Ratio: {token_set_similarity}")

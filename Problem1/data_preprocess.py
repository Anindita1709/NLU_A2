import os
import re
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#  path of the file

txt_files = [
    "data/iitj_departments.txt",
    "data/CSE_syllabus.txt",
    "data/ACAD_regulation.txt",
    "data/ACAD_calendar.txt",
    "data/iitj_home_page.txt",
    "data/iitj_corpus.txt",
]


# configuration

BOILERPLATE_PATTERNS = [
    r"redirecttologinpage",       #detection of boilerplate text (pattern)
    r"sitemap",
    r"important links.*",
    r"copyright.*",
    r"all rights reserved.*",
    r"feedback.*",
    r"web policy.*",
    r"web information manager.*",
    r"for any comments.*",
    r"old website.*",
    r"contact old website.*",
    r"arrow downward",
    r"iit jodhpur",
    r"indian institute of technology jodhpur",
    r"n\.?h\.?\s*62.*jodhpur.*",
    r"https?://\S+",
    r"\S+@\S+",
]

# stopwords for generating better word cloud
EXTRA_STOPWORDS = {
    "the","is","are","was","were","am","be","been","being",
    "of","to","in","on","at","for","from","by","with","as",
    "and","or","an","a","this","that","these","those","it",
    "its","their","his","her","them","they","he","she","we",
    "you","your","our","i","me","my","us",
    "will","shall","may","can","could","would","should",
    "have","has","had","do","does","did","done",
    "not","if","than","then","there","here","such","all","any",
    "also","into","under","over","per","during","within",
    "after","before","about","between","through","same",
    "each","every","either","neither","other","others",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "www","http","https",
    "iit","iitj","iitjodhpur","jodhpur",
    "academic","course","courses","student","students","semester",
    "program","programme","day","days","date","year","years",
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
    "mon","tue","wed","thu","fri","sat","sun"
}


#extract text from files
def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def remove_boilerplate(text):       #function for removal of boilerplate text and formatting artifacts
    text = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


#removal of excessive punctuation and non-textual content

def preprocess_text(text):              
    text = remove_boilerplate(text)

    # normalize dashes and symbols
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    text = text.replace("&", " and ")

    # remove non-ascii
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # remove numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # keep only alphabets and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # tokenization
    tokens = re.findall(r"[a-z]+", text)

    # remove very short junk tokens
    tokens = [tok for tok in tokens if len(tok) > 1]

    return tokens

# reading text file and preprocessing
raw_docs = {}
token_docs = {}

for path in txt_files:
    name = os.path.basename(path)
    raw_text = extract_text_from_txt(path)

    if raw_text is None:
        print(f"Skipping file: {name}")
        continue

    tokens = preprocess_text(raw_text)

    raw_docs[name] = raw_text
    token_docs[name] = tokens


# counting frequency of data
all_tokens = [tok for toks in token_docs.values() for tok in toks]
vocab = sorted(set(all_tokens))
freq = Counter(all_tokens)

stats = {
    "total_documents": len(token_docs),
    "total_tokens": len(all_tokens),
    "vocabulary_size": len(vocab),
    "per_document_stats": {
        name: {
            "tokens": len(tokens),
            "vocabulary_size": len(set(tokens)),
        }
        for name, tokens in token_docs.items()
    },
    "top_10_tokens": freq.most_common(10),
}

with open("dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

#  save cleaned corpus
with open("corpus.txt", "w", encoding="utf-8") as f:
    for name, tokens in token_docs.items():
        f.write(f"# DOCUMENT: {name}\n")
        f.write(" ".join(tokens))
        f.write("\n\n")

# csv file containing data statistic per document
rows = []
for name, tokens in token_docs.items():
    rows.append({
        "document": name,
        "num_tokens": len(tokens),
        "vocabulary_size": len(set(tokens))
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv("per_document_stats.csv", index=False)


# generation of word cloud after removing stopwords

wc_stopwords = set(STOPWORDS).union(EXTRA_STOPWORDS)
wc_freq = {w: c for w, c in freq.items() if w not in wc_stopwords}

wordcloud = WordCloud(
    width=1400,
    height=800,
    background_color="white",
    stopwords=wc_stopwords,
    collocations=False,
).generate_from_frequencies(wc_freq)

plt.figure(figsize=(14, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("wordcloud.png", dpi=200, bbox_inches="tight")
plt.show()

#printing the results
print("Total documents:", len(token_docs))
print("Total tokens:", len(all_tokens))
print("Vocabulary size:", len(vocab))

print("\nPer-document stats:")
for name, tokens in token_docs.items():
    print(f"{name}: tokens={len(tokens)}, vocab={len(set(tokens))}")

print("\nSaved files:")
print("- cleaned_corpus.txt")
print("- dataset_stats.json")
print("- per_document_stats.csv")
print("- wordcloud.png")

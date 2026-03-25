import re
import random
import json
import math
from collections import Counter

import numpy as np


# LOAD AND PREPARE DATA


def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    # remove document headers like "# DOCUMENT: ..."
    text = re.sub(r"# document:.*", " ", text)

    # keep only words
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def build_vocab(tokens, min_count=1):
    word_freq = Counter(tokens)
    vocab = [w for w, c in word_freq.items() if c >= min_count]

    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}

    filtered_tokens = [w for w in tokens if w in word_to_id]
    counts = np.array([word_freq[w] for w in vocab], dtype=np.float64)

    return vocab, word_to_id, id_to_word, filtered_tokens, word_freq, counts


# CREATE TRAINING PAIRS for CBOW & SKIPGRAM


def generate_cbow_data(tokens, word_to_id, window_size=2):
    data = []
    n = len(tokens)

    for i in range(n):
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        context = []
        for j in range(left, right):
            if j != i:
                context.append(word_to_id[tokens[j]])

        if len(context) > 0:
            target = word_to_id[tokens[i]]
            data.append((context, target))

    return data


def generate_skipgram_data(tokens, word_to_id, window_size=2):
    data = []
    n = len(tokens)

    for i in range(n):
        center = word_to_id[tokens[i]]
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        for j in range(left, right):
            if j != i:
                context = word_to_id[tokens[j]]
                data.append((center, context))

    return data

#sigmoid func.
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

#measure cosine similarity between words
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return np.dot(a, b) / denom

#unigram distribution of words sequence 
def build_unigram_distribution(counts, power=0.75):
    dist = counts ** power
    dist = dist / dist.sum()
    return dist

#negative sampling
def sample_negative_ids(vocab_size, positive_id, num_negatives, unigram_dist):
    negatives = []
    while len(negatives) < num_negatives:
        neg = np.random.choice(vocab_size, p=unigram_dist)
        if neg != positive_id:
            negatives.append(int(neg))
    return negatives

#detect top 5 nearest neighbour of words based on cosine similarity

def nearest_neighbors(word, embeddings, word_to_id, id_to_word, top_k=5):
    if word not in word_to_id:
        return []

    query_id = word_to_id[word]
    query_vec = embeddings[query_id]

    sims = []
    for i in range(len(embeddings)):
        if i == query_id:
            continue
        sim = cosine_similarity(query_vec, embeddings[i])
        sims.append((id_to_word[i], float(sim)))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

#drawing analogy between results

def analogy(a, b, c, embeddings, word_to_id, id_to_word, top_k=5):
    if a not in word_to_id or b not in word_to_id or c not in word_to_id:
        return []

    vec = embeddings[word_to_id[b]] - embeddings[word_to_id[a]] + embeddings[word_to_id[c]]

    excluded = {word_to_id[a], word_to_id[b], word_to_id[c]}
    sims = []

    for i in range(len(embeddings)):
        if i in excluded:
            continue
        sim = cosine_similarity(vec, embeddings[i])
        sims.append((id_to_word[i], float(sim)))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# MANUAL CBOW

class CBOWWord2Vec:
    def __init__(self, vocab_size, embedding_dim, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def train(self, data, epochs=3, lr=0.025, num_negatives=5, unigram_dist=None, verbose=True):
        for epoch in range(epochs):
            random.shuffle(data)
            total_loss = 0.0

            for context_ids, target_id in data:
                # average context embeddings
                h = np.mean(self.W_in[context_ids], axis=0)

                # positive example
                u_pos = self.W_out[target_id]
                score_pos = sigmoid(np.dot(u_pos, h))
                loss = -math.log(score_pos + 1e-10)

                grad_h = (score_pos - 1.0) * u_pos
                grad_u_pos = (score_pos - 1.0) * h

                self.W_out[target_id] -= lr * grad_u_pos

                # negative samples
                negative_ids = sample_negative_ids(
                    self.vocab_size, target_id, num_negatives, unigram_dist
                )

                for neg_id in negative_ids:
                    u_neg = self.W_out[neg_id]
                    score_neg = sigmoid(np.dot(u_neg, h))
                    loss += -math.log(1.0 - score_neg + 1e-10)

                    grad_h += score_neg * u_neg
                    grad_u_neg = score_neg * h

                    self.W_out[neg_id] -= lr * grad_u_neg

                # backprop to input embeddings
                grad_context = grad_h / len(context_ids)
                for cid in context_ids:
                    self.W_in[cid] -= lr * grad_context

                total_loss += loss

            avg_loss = total_loss / len(data)
            if verbose:
                print(f"[CBOW] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    def get_embeddings(self):
        return self.W_in


#  MANUAL SKIP-GRAM

class SkipGramWord2Vec:
    def __init__(self, vocab_size, embedding_dim, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def train(self, data, epochs=3, lr=0.025, num_negatives=5, unigram_dist=None, verbose=True):
        for epoch in range(epochs):
            random.shuffle(data)
            total_loss = 0.0

            for center_id, context_id in data:
                v = self.W_in[center_id]

                # positive example
                u_pos = self.W_out[context_id]
                score_pos = sigmoid(np.dot(u_pos, v))
                loss = -math.log(score_pos + 1e-10)

                grad_v = (score_pos - 1.0) * u_pos
                grad_u_pos = (score_pos - 1.0) * v

                self.W_out[context_id] -= lr * grad_u_pos

                # negative samples
                negative_ids = sample_negative_ids(
                    self.vocab_size, context_id, num_negatives, unigram_dist
                )

                for neg_id in negative_ids:
                    u_neg = self.W_out[neg_id]
                    score_neg = sigmoid(np.dot(u_neg, v))
                    loss += -math.log(1.0 - score_neg + 1e-10)

                    grad_v += score_neg * u_neg
                    grad_u_neg = score_neg * v

                    self.W_out[neg_id] -= lr * grad_u_neg

                self.W_in[center_id] -= lr * grad_v
                total_loss += loss

            avg_loss = total_loss / len(data)
            if verbose:
                print(f"[Skip-gram] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    def get_embeddings(self):
        return self.W_in


#  EVALUATION OF NEAREST NEIGHBOUR,ANALOGY EXPERIMENT

def evaluate_model(model_name, embeddings, word_to_id, id_to_word):

    print(f"{model_name} - Nearest Neighbors")
    print("\n" +"-" * 60)

    query_words = ["research", "student", "phd", "exam"]

    for word in query_words:
        if word in word_to_id:
            print(f"\nTop 5 nearest neighbors for '{word}':")
            for w, score in nearest_neighbors(word, embeddings, word_to_id, id_to_word, top_k=5):
                print(f"  {w:15s} {score:.4f}")
        else:
            print(f"\n'{word}' not found in vocabulary.")

    print(f"{model_name} - Analogy Experiments")
    print("\n" +"-" * 60)

    analogies = [
        ("ug", "btech", "pg"),
        ("student", "exam", "faculty"),
        ("research", "lab", "teaching"),
    ]

    for a, b, c in analogies:
        print(f"\n{a} : {b} :: {c} : ?")
        results = analogy(a, b, c, embeddings, word_to_id, id_to_word, top_k=5)
        if results:
            for w, score in results:
                print(f"  {w:15s} {score:.4f}")
        else:
            print("  Analogy could not be computed because one or more words are missing.")

#evaluate the embedding

def score_queries(embeddings, word_to_id, id_to_word):
    """
    Simple proxy metric for comparing experiments.
    Uses average cosine similarity among meaningful academic pairs if present.
    """
    test_pairs = [
        ("research", "phd"),
        ("student", "exam"),
        ("btech", "ug"),
        ("mtech", "pg"),
    ]

    sims = []
    for w1, w2 in test_pairs:
        if w1 in word_to_id and w2 in word_to_id:
            v1 = embeddings[word_to_id[w1]]
            v2 = embeddings[word_to_id[w2]]
            sims.append(cosine_similarity(v1, v2))

    if len(sims) == 0:
        return None
    return float(np.mean(sims))


# MAIN EXPERIMENT LOOP


def run_experiments(corpus_path="corpus.txt"):
    tokens = read_corpus(corpus_path)
    print("Total raw tokens:", len(tokens))

    vocab, word_to_id, id_to_word, filtered_tokens, word_freq, counts = build_vocab(tokens, min_count=1)
    vocab_size = len(vocab)
    unigram_dist = build_unigram_distribution(counts, power=0.75)

    print("Vocabulary size:", vocab_size)

    experiment_results = []

    embedding_dims = [50, 100]
    window_sizes = [2, 4]
    negative_samples_list = [3, 5]

    best_cbow = None
    best_skipgram = None
    best_cbow_score = -1e9
    best_skipgram_score = -1e9

    for embedding_dim in embedding_dims:
        for window_size in window_sizes:
            print("\n" + "#" * 70)
            print(f"Preparing data for window size = {window_size}")
            print("#" * 70)

            cbow_data = generate_cbow_data(filtered_tokens, word_to_id, window_size=window_size)
            skipgram_data = generate_skipgram_data(filtered_tokens, word_to_id, window_size=window_size)

            print(f"CBOW pairs: {len(cbow_data)}")
            print(f"Skip-gram pairs: {len(skipgram_data)}")

            for num_negatives in negative_samples_list:
                print("\n" + "-" * 70)
                print(f"Embedding dim = {embedding_dim}, Window = {window_size}, Negatives = {num_negatives}")
                print("-" * 70)

                # CBOW
                cbow_model = CBOWWord2Vec(vocab_size, embedding_dim, seed=42)
                cbow_model.train(
                    cbow_data,
                    epochs=3,
                    lr=0.03,
                    num_negatives=num_negatives,
                    unigram_dist=unigram_dist,
                    verbose=True
                )
                cbow_embeddings = cbow_model.get_embeddings()
                cbow_score = score_queries(cbow_embeddings, word_to_id, id_to_word)

                # Skip-gram
                skip_model = SkipGramWord2Vec(vocab_size, embedding_dim, seed=42)
                skip_model.train(
                    skipgram_data,
                    epochs=3,
                    lr=0.03,
                    num_negatives=num_negatives,
                    unigram_dist=unigram_dist,
                    verbose=True
                )
                skip_embeddings = skip_model.get_embeddings()
                skip_score = score_queries(skip_embeddings, word_to_id, id_to_word)

                experiment_results.append({
                    "embedding_dim": embedding_dim,
                    "window_size": window_size,
                    "num_negatives": num_negatives,
                    "cbow_score": cbow_score,
                    "skipgram_score": skip_score
                })

                if cbow_score is not None and cbow_score > best_cbow_score:
                    best_cbow_score = cbow_score
                    best_cbow = {
                        "model": cbow_model,
                        "embeddings": cbow_embeddings,
                        "embedding_dim": embedding_dim,
                        "window_size": window_size,
                        "num_negatives": num_negatives
                    }

                if skip_score is not None and skip_score > best_skipgram_score:
                    best_skipgram_score = skip_score
                    best_skipgram = {
                        "model": skip_model,
                        "embeddings": skip_embeddings,
                        "embedding_dim": embedding_dim,
                        "window_size": window_size,
                        "num_negatives": num_negatives
                    }

    # Save results
    with open("manual_word2vec_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print("\n" + "-" * 70)
    print("EXPERIMENT SUMMARY")
    print("-" * 70)
    for row in experiment_results:
        print(row)

    # Evaluate best CBOW
    if best_cbow is not None:
        print("\n" + "-" * 70)
        print("BEST CBOW CONFIGURATION")
        print("-" * 70)
        print({
            "embedding_dim": best_cbow["embedding_dim"],
            "window_size": best_cbow["window_size"],
            "num_negatives": best_cbow["num_negatives"],
            "score": best_cbow_score
        })
        evaluate_model("CBOW", best_cbow["embeddings"], word_to_id, id_to_word)

    # Evaluate best Skip-gram
    if best_skipgram is not None:
        print("\n" + "-" * 70)
        print("BEST SKIP-GRAM CONFIGURATION")
        print("-" * 70)
        print({
            "embedding_dim": best_skipgram["embedding_dim"],
            "window_size": best_skipgram["window_size"],
            "num_negatives": best_skipgram["num_negatives"],
            "score": best_skipgram_score
        })
        evaluate_model("Skip-gram", best_skipgram["embeddings"], word_to_id, id_to_word)

    return experiment_results

# RUN


if __name__ == "__main__":
    run_experiments("corpus.txt")
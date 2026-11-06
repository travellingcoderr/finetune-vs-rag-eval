"""Non-LLM baselines. Every LLM approach should beat these before it earns its cost.

  tfidf_logreg  TF-IDF word+char n-grams -> logistic regression (trained on the full train split)
  embed_logreg  all-MiniLM-L6-v2 sentence embeddings -> logistic regression
  embed_knn     nearest labeled neighbor by embedding (this is the retriever the RAG variant uses)

python src/baselines.py --limit 300
"""
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from common import DATA, append_result, cost_per_1k, percentile, read_jsonl, scores

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=300)
p.add_argument("--usd-per-hour", type=float, default=0.0892, help="CPU instance price, default AWS c7i.large (2 vCPU) on-demand us-east-1, Nov 2026")
p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
a = p.parse_args()

train = read_jsonl(DATA / "train.jsonl")
test = read_jsonl(DATA / "golden_test.jsonl", a.limit)
Xtr, ytr = [r["text"] for r in train], [r["label"] for r in train]
Xte, yte = [r["text"] for r in test], [r["label"] for r in test]


def timed_predict(fn, xs):
    preds, lat = [], []
    for x in xs:  # one at a time, to compare latency fairly with the LLM runs
        t0 = time.perf_counter()
        preds.append(fn(x))
        lat.append(time.perf_counter() - t0)
    return preds, lat


def record(name, model, preds, lat, train_s):
    r = {"run": name, "approach": "classical", "model": model, "n": len(yte), **scores(yte, preds),
         "p50_latency_s": round(percentile(lat, 0.5), 4), "p95_latency_s": round(percentile(lat, 0.95), 4),
         "usd_per_1k": cost_per_1k(lat, a.usd_per_hour, 0, 0, None, None), "train_s": round(train_s, 1)}
    print(r)
    append_result(r)


t0 = time.perf_counter()
tfidf = make_pipeline(
    make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
               TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    LogisticRegression(max_iter=2000, C=10),
).fit(Xtr, ytr)
preds, lat = timed_predict(lambda x: tfidf.predict([x])[0], Xte)
record("tfidf_logreg", "tfidf+logreg", preds, lat, time.perf_counter() - t0)

from sentence_transformers import SentenceTransformer  # noqa: E402

emb = SentenceTransformer(a.embedder, device="cpu")
t0 = time.perf_counter()
Etr = emb.encode(Xtr, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
embed_s = time.perf_counter() - t0
clf = LogisticRegression(max_iter=3000, C=20).fit(Etr, ytr)
logreg_s = time.perf_counter() - t0
ytr_arr = np.array(ytr)
preds, lat = timed_predict(lambda x: clf.predict(emb.encode([x], normalize_embeddings=True))[0], Xte)
record("embed_logreg", a.embedder.split("/")[-1] + "+logreg", preds, lat, logreg_s)
preds, lat = timed_predict(lambda x: ytr_arr[int(np.argmax(Etr @ emb.encode([x], normalize_embeddings=True)[0]))], Xte)
record("embed_knn", a.embedder.split("/")[-1] + "+1nn", preds, lat, embed_s)

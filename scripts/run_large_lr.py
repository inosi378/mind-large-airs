import argparse
import csv
import gc
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

Behavior = Tuple[str, List[str], List[Tuple[str, int]]]


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value == "nan":
        return ""
    return value


def read_news(path: Path) -> Dict[str, Dict[str, str]]:
    news: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.rstrip("\n").split("\t")
            if len(row) < 8:
                row += [""] * (8 - len(row))
            if len(row) > 8:
                row = row[:8]

            item = dict(zip(NEWS_COLUMNS, row))
            news_id = item["news_id"]
            title = normalize_text(item["title"])
            abstract = normalize_text(item["abstract"])
            text = f"{title} {abstract}".strip()
            if not text:
                text = title

            news[news_id] = {
                "category": normalize_text(item["category"]),
                "subcategory": normalize_text(item["subcategory"]),
                "title": title,
                "abstract": abstract,
                "text": text,
            }
    return news


def parse_behavior_line(line: str) -> Optional[Behavior]:
    row = line.rstrip("\n").split("\t")
    if len(row) != 5:
        return None

    impression_id, user_id, time_str, history_str, impressions_str = row
    history = history_str.split() if history_str else []

    candidates: List[Tuple[str, int]] = []
    for token in impressions_str.split():
        if "-" not in token:
            continue
        news_id, label_str = token.rsplit("-", 1)
        if label_str not in {"0", "1"}:
            continue
        candidates.append((news_id, int(label_str)))

    if not candidates:
        return None

    return impression_id, history, candidates


def read_behaviors(path: Path, max_lines: Optional[int] = None) -> List[Behavior]:
    behaviors: List[Behavior] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            parsed = parse_behavior_line(line)
            if parsed is not None:
                behaviors.append(parsed)
    return behaviors


def collect_needed_news_ids(behaviors: Iterable[Behavior]) -> Tuple[set, int]:
    needed = set()
    pair_count = 0
    for _impression_id, history, candidates in behaviors:
        needed.update(history)
        for news_id, _label in candidates:
            needed.add(news_id)
            pair_count += 1
    return needed, pair_count


def choose_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def fit_tfidf(
    train_news: Dict[str, Dict[str, str]],
    train_needed_ids: set,
    max_features: int,
    min_df: int,
) -> TfidfVectorizer:
    fit_ids = sorted(nid for nid in train_needed_ids if nid in train_news)
    texts = [train_news[nid]["text"] for nid in fit_ids]
    if not texts:
        raise ValueError("No train news texts available for TF-IDF fitting.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 1),
        stop_words="english",
        lowercase=True,
        norm="l2",
        dtype=np.float32,
    )
    vectorizer.fit(texts)
    return vectorizer


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    news: Dict[str, Dict[str, str]],
    needed_ids: set,
) -> Tuple[List[str], sparse.csr_matrix, Dict[str, int]]:
    ids = sorted(nid for nid in needed_ids if nid in news)
    texts = [news[nid]["text"] for nid in ids]
    matrix = vectorizer.transform(texts).tocsr()
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    return ids, matrix, id_to_idx


def cache_key(ids: List[str], model_name: str) -> str:
    raw = model_name + "\n" + "\n".join(ids)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def encode_embeddings(
    news: Dict[str, Dict[str, str]],
    needed_ids: set,
    model_name: str,
    cache_dir: Path,
    split_name: str,
    batch_size: int,
    device: str,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    ids = sorted(nid for nid in needed_ids if nid in news)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    key = cache_key(ids, model_name)
    cache_path = cache_dir / f"{split_name}_{safe_model_name}_{key}.npz"

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=False)
        cached_ids = loaded["ids"].astype(str).tolist()
        emb = loaded["embeddings"].astype(np.float32)
        if cached_ids != ids:
            raise ValueError(f"Embedding cache id mismatch: {cache_path}")
        return ids, emb, id_to_idx

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    texts = [news[nid]["text"] for nid in ids]

    print(f"Encoding {split_name} embeddings: {len(ids):,} news")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(ids), embeddings=emb)
    print(f"Saved embedding cache: {cache_path}")
    return ids, emb, id_to_idx


def sparse_user_weights(tfidf_matrix: sparse.csr_matrix, hist_indices: List[int]) -> Optional[Dict[int, float]]:
    if not hist_indices:
        return None

    accum: Dict[int, float] = {}
    denom = float(len(hist_indices))
    for idx in hist_indices:
        row = tfidf_matrix.getrow(idx)
        for col, val in zip(row.indices, row.data):
            accum[int(col)] = accum.get(int(col), 0.0) + float(val) / denom

    if not accum:
        return None

    norm = math.sqrt(sum(v * v for v in accum.values()))
    if norm <= 0:
        return None

    for col in list(accum.keys()):
        accum[col] /= norm
    return accum


def sparse_candidate_score(
    tfidf_matrix: sparse.csr_matrix,
    cand_idx: int,
    user_weights: Optional[Dict[int, float]],
) -> float:
    if user_weights is None:
        return 0.0
    row = tfidf_matrix.getrow(cand_idx)
    if row.nnz == 0:
        return 0.0
    return float(sum(float(val) * user_weights.get(int(col), 0.0) for col, val in zip(row.indices, row.data)))


def normalized_mean_embedding(
    emb_matrix: np.ndarray,
    hist_indices: List[int],
) -> Optional[np.ndarray]:
    if not hist_indices:
        return None
    user_vec = emb_matrix[hist_indices].mean(axis=0)
    norm = np.linalg.norm(user_vec)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return (user_vec / norm).astype(np.float32)


def embedding_candidate_score(
    emb_matrix: np.ndarray,
    cand_idx: int,
    user_vec: Optional[np.ndarray],
) -> float:
    if user_vec is None:
        return 0.0
    return float(np.dot(emb_matrix[cand_idx], user_vec))


def category_pref_scores(
    news: Dict[str, Dict[str, str]],
    history: List[str],
) -> Tuple[Counter, Counter, int, int]:
    category_counter = Counter()
    subcategory_counter = Counter()
    category_total = 0
    subcategory_total = 0

    for hist_id in history:
        item = news.get(hist_id)
        if item is None:
            continue
        category = item.get("category", "")
        subcategory = item.get("subcategory", "")
        if category:
            category_counter[category] += 1
            category_total += 1
        if subcategory:
            subcategory_counter[subcategory] += 1
            subcategory_total += 1

    return category_counter, subcategory_counter, category_total, subcategory_total


def build_feature_table(
    behaviors: List[Behavior],
    news: Dict[str, Dict[str, str]],
    tfidf_matrix: sparse.csr_matrix,
    tfidf_id_to_idx: Dict[str, int],
    emb_matrix: np.ndarray,
    emb_id_to_idx: Dict[str, int],
    expected_pairs: int,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, str]]]:
    X = np.zeros((expected_pairs, 4), dtype=np.float32)
    y = np.zeros(expected_pairs, dtype=np.int8)
    group_ranges: List[Tuple[int, int, str]] = []

    row_idx = 0
    for impression_id, history, candidates in tqdm(behaviors, desc=desc):
        start = row_idx

        cat_counter, subcat_counter, cat_total, subcat_total = category_pref_scores(news, history)

        hist_tfidf_indices = [tfidf_id_to_idx[nid] for nid in history if nid in tfidf_id_to_idx]
        user_tfidf = sparse_user_weights(tfidf_matrix, hist_tfidf_indices)

        hist_emb_indices = [emb_id_to_idx[nid] for nid in history if nid in emb_id_to_idx]
        user_emb = normalized_mean_embedding(emb_matrix, hist_emb_indices)

        for cand_id, label in candidates:
            item = news.get(cand_id)
            if item is None:
                category_pref = 0.0
                subcategory_pref = 0.0
            else:
                category = item.get("category", "")
                subcategory = item.get("subcategory", "")
                category_pref = float(cat_counter.get(category, 0) / cat_total) if category and cat_total > 0 else 0.0
                subcategory_pref = float(subcat_counter.get(subcategory, 0) / subcat_total) if subcategory and subcat_total > 0 else 0.0

            tfidf_score = 0.0
            if cand_id in tfidf_id_to_idx:
                tfidf_score = sparse_candidate_score(tfidf_matrix, tfidf_id_to_idx[cand_id], user_tfidf)

            embedding_score = 0.0
            if cand_id in emb_id_to_idx:
                embedding_score = embedding_candidate_score(emb_matrix, emb_id_to_idx[cand_id], user_emb)

            X[row_idx, 0] = category_pref
            X[row_idx, 1] = subcategory_pref
            X[row_idx, 2] = tfidf_score
            X[row_idx, 3] = embedding_score
            y[row_idx] = label
            row_idx += 1

        end = row_idx
        if end > start:
            group_ranges.append((start, end, impression_id))

    if row_idx != expected_pairs:
        X = X[:row_idx]
        y = y[:row_idx]

    return X, y, group_ranges


def dcg_at_k(labels: np.ndarray, k: int) -> float:
    labels = labels[:k]
    if labels.size == 0:
        return 0.0
    gains = (2 ** labels - 1).astype(np.float64)
    discounts = np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(labels_sorted_by_score: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(labels_sorted_by_score, k)
    ideal = np.sort(labels_sorted_by_score)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ranges: List[Tuple[int, int, str]],
) -> Dict[str, float]:
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for start, end, _impression_id in group_ranges:
        labels = y_true[start:end].astype(np.int32)
        scores = y_score[start:end]
        if labels.size == 0:
            continue

        if len(np.unique(labels)) == 2:
            aucs.append(float(roc_auc_score(labels, scores)))

        order = np.argsort(-scores)
        ranked_labels = labels[order]

        pos = np.where(ranked_labels == 1)[0]
        mrrs.append(float(1.0 / (pos[0] + 1)) if pos.size > 0 else 0.0)
        ndcg5s.append(ndcg_at_k(ranked_labels, 5))
        ndcg10s.append(ndcg_at_k(ranked_labels, 10))

    return {
        "AUC": float(np.mean(aucs)) if aucs else float("nan"),
        "MRR": float(np.mean(mrrs)) if mrrs else float("nan"),
        "nDCG@5": float(np.mean(ndcg5s)) if ndcg5s else float("nan"),
        "nDCG@10": float(np.mean(ndcg10s)) if ndcg10s else float("nan"),
        "valid_auc_groups": int(len(aucs)),
        "groups": int(len(group_ranges)),
    }


def append_result_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def positive_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dev_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tfidf_max_features", type=int, default=50000)
    parser.add_argument("--tfidf_min_df", type=int, default=2)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_train_behaviors", type=int, default=None)
    parser.add_argument("--max_dev_behaviors", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", type=str, default="none", choices=["none", "balanced"])
    args = parser.parse_args()

    started = time.time()
    train_dir = Path(args.train_dir)
    dev_dir = Path(args.dev_dir)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIND-large LR ranking experiment")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    train_news_path = train_dir / "news.tsv"
    train_behaviors_path = train_dir / "behaviors.tsv"
    dev_news_path = dev_dir / "news.tsv"
    dev_behaviors_path = dev_dir / "behaviors.tsv"

    for path in [train_news_path, train_behaviors_path, dev_news_path, dev_behaviors_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("\n[1] Loading news")
    train_news = read_news(train_news_path)
    dev_news = read_news(dev_news_path)
    print(f"train news: {len(train_news):,}")
    print(f"dev news  : {len(dev_news):,}")

    print("\n[2] Loading behaviors")
    train_behaviors = read_behaviors(train_behaviors_path, args.max_train_behaviors)
    dev_behaviors = read_behaviors(dev_behaviors_path, args.max_dev_behaviors)
    print(f"train behaviors: {len(train_behaviors):,}")
    print(f"dev behaviors  : {len(dev_behaviors):,}")

    train_needed, train_pairs = collect_needed_news_ids(train_behaviors)
    dev_needed, dev_pairs = collect_needed_news_ids(dev_behaviors)
    print(f"train candidate rows: {train_pairs:,}")
    print(f"dev candidate rows  : {dev_pairs:,}")
    print(f"train needed news ids: {len(train_needed):,}")
    print(f"dev needed news ids  : {len(dev_needed):,}")

    print("\n[3] Building TF-IDF")
    t0 = time.time()
    vectorizer = fit_tfidf(train_news, train_needed, args.tfidf_max_features, args.tfidf_min_df)
    train_tfidf_ids, train_tfidf, train_tfidf_idx = transform_tfidf(vectorizer, train_news, train_needed)
    dev_tfidf_ids, dev_tfidf, dev_tfidf_idx = transform_tfidf(vectorizer, dev_news, dev_needed)
    tfidf_time = time.time() - t0
    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"train TF-IDF shape: {train_tfidf.shape}")
    print(f"dev TF-IDF shape  : {dev_tfidf.shape}")
    print(f"TF-IDF time: {tfidf_time:.1f} sec")

    print("\n[4] Building embeddings")
    t0 = time.time()
    device = choose_device(args.device)
    train_emb_ids, train_emb, train_emb_idx = encode_embeddings(
        train_news,
        train_needed,
        args.embedding_model,
        cache_dir,
        "train",
        args.embedding_batch_size,
        device,
    )
    dev_emb_ids, dev_emb, dev_emb_idx = encode_embeddings(
        dev_news,
        dev_needed,
        args.embedding_model,
        cache_dir,
        "dev",
        args.embedding_batch_size,
        device,
    )
    emb_time = time.time() - t0
    print(f"train embedding shape: {train_emb.shape}")
    print(f"dev embedding shape  : {dev_emb.shape}")
    print(f"Embedding time: {emb_time:.1f} sec")

    print("\n[5] Building feature tables")
    t0 = time.time()
    X_train, y_train, train_groups = build_feature_table(
        train_behaviors,
        train_news,
        train_tfidf,
        train_tfidf_idx,
        train_emb,
        train_emb_idx,
        train_pairs,
        "train features",
    )
    X_dev, y_dev, dev_groups = build_feature_table(
        dev_behaviors,
        dev_news,
        dev_tfidf,
        dev_tfidf_idx,
        dev_emb,
        dev_emb_idx,
        dev_pairs,
        "dev features",
    )
    feature_time = time.time() - t0
    print(f"X_train: {X_train.shape}, positive rate: {positive_rate(y_train):.6f}")
    print(f"X_dev  : {X_dev.shape}, positive rate: {positive_rate(y_dev):.6f}")
    print(f"Feature time: {feature_time:.1f} sec")

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_dev.npy", X_dev)
    np.save(out_dir / "y_dev.npy", y_dev)

    print("\n[6] Training Logistic Regression")
    t0 = time.time()
    class_weight = None if args.class_weight == "none" else "balanced"
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    C=args.C,
                    max_iter=args.max_iter,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f} sec")

    joblib.dump(model, out_dir / "lr_model.joblib")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")

    print("\n[7] Evaluating")
    dev_scores = model.predict_proba(X_dev)[:, 1]
    metrics = evaluate_ranking(y_dev, dev_scores, dev_groups)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    lr = model.named_steps["lr"]
    coef = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    feature_names = ["category_pref", "subcategory_pref", "tfidf_score", "embedding_score"]
    coef_info = {name: float(value) for name, value in zip(feature_names, coef)}
    print("\nLR coefficients after scaling:")
    print(json.dumps(coef_info, indent=2, ensure_ascii=False))
    print(f"intercept: {intercept:.6f}")

    total_time = time.time() - started
    result = {
        "dataset": "MIND-large",
        "train_dir": str(train_dir),
        "dev_dir": str(dev_dir),
        "train_behaviors": len(train_behaviors),
        "dev_behaviors": len(dev_behaviors),
        "train_rows": int(X_train.shape[0]),
        "dev_rows": int(X_dev.shape[0]),
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "embedding_model": args.embedding_model,
        "AUC": metrics["AUC"],
        "MRR": metrics["MRR"],
        "nDCG@5": metrics["nDCG@5"],
        "nDCG@10": metrics["nDCG@10"],
        "tfidf_time_sec": round(tfidf_time, 3),
        "embedding_time_sec": round(emb_time, 3),
        "feature_time_sec": round(feature_time, 3),
        "train_time_sec": round(train_time, 3),
        "total_time_sec": round(total_time, 3),
        "coef_category_pref": coef_info["category_pref"],
        "coef_subcategory_pref": coef_info["subcategory_pref"],
        "coef_tfidf_score": coef_info["tfidf_score"],
        "coef_embedding_score": coef_info["embedding_score"],
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "result": result, "coef": coef_info}, f, indent=2, ensure_ascii=False)

    append_result_csv(out_dir / "results.csv", result)
    print(f"\nSaved outputs to: {out_dir}")
    print(f"Total time: {total_time:.1f} sec")

    gc.collect()


if __name__ == "__main__":
    main()

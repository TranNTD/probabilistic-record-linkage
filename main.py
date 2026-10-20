"""Privacy-safe, reproducible probabilistic record linkage demo.

Run: python main.py
All records are synthetic; no external data is downloaded.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.metrics import precision_recall_fscore_support

SEED = 42
OUT = Path(__file__).parent / "outputs"


def generate_data(n: int = 350, seed: int = SEED):
    rng = np.random.default_rng(seed)
    first = np.array(["Anna", "Lena", "Emma", "Mia", "Noah", "Paul", "Jonas", "Leon", "Sarah", "Marie", "Max", "Sofia"])
    last = np.array(["Mueller", "Schmidt", "Weber", "Fischer", "Wagner", "Becker", "Hoffmann", "Schneider", "Meyer", "Schulz"])
    streets = np.array(["Hauptstrasse", "Bahnhofstrasse", "Gartenweg", "Berliner Strasse", "Parkweg"])
    rows = []
    for i in range(n):
        year = int(rng.integers(1960, 2005))
        rows.append({"person_id": i, "first_name": str(rng.choice(first)),
                     "last_name": str(rng.choice(last)), "birth_year": year,
                     "postcode": f"{int(rng.integers(10000, 99999)):05d}",
                     "street": f"{str(rng.choice(streets))} {int(rng.integers(1, 151))}"})
    a = pd.DataFrame(rows)
    a.insert(0, "record_id", [f"A{i:04d}" for i in range(n)])
    
    # Only 80% of A persons appear in B, and B contains new people.
    kept = rng.choice(n, size=int(n * .8), replace=False)
    b_rows = a.iloc[kept].copy().reset_index(drop=True)
    for i in range(len(b_rows)):
        if rng.random() < .17:
            column = rng.choice(["first_name", "last_name"])
            s = b_rows.at[i, column]
            if len(s) > 3:
                pos = int(rng.integers(0, len(s) - 1))
                b_rows.at[i, column] = s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]
        if rng.random() < .12:
            b_rows.at[i, "postcode"] = str(b_rows.at[i, "postcode"])[:4] + str(int(rng.integers(0, 10)))
        if rng.random() < .15:
            b_rows.at[i, "street"] = b_rows.at[i, "street"].replace("strasse", "str.")
    extra_rows = []
    for j in range(int(.12 * n)):
        extra_rows.append({"record_id": "", "person_id": n + j,
                           "first_name": str(rng.choice(first)), "last_name": str(rng.choice(last)),
                           "birth_year": int(rng.integers(1960, 2005)),
                           "postcode": f"{int(rng.integers(10000, 99999)):05d}",
                           "street": f"{str(rng.choice(streets))} {int(rng.integers(1,151))}"})
    b = pd.concat([b_rows, pd.DataFrame(extra_rows)], ignore_index=True)
    b["record_id"] = [f"B{i:04d}" for i in range(len(b))]
    return a, b


def candidates(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    # Blocking avoids a full Cartesian product. Year is stable in synthetic data.
    left = a.assign(block=a.birth_year.astype(str))
    right = b.assign(block=b.birth_year.astype(str))
    pairs = left.merge(right, on="block", suffixes=("_a", "_b"))
    return pairs


def compare(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    for col in ("first_name", "last_name", "street"):
        similarity = [ratio(str(x).lower(), str(y).lower()) / 100
                      for x, y in zip(pairs[f"{col}_a"], pairs[f"{col}_b"])]
        pairs[f"agree_{col}"] = (np.array(similarity) >= ({"first_name": .82, "last_name": .82, "street": .75}[col])).astype(int)
    pairs["agree_postcode"] = (pairs.postcode_a == pairs.postcode_b).astype(int)
    pairs["agree_birth_year"] = (pairs.birth_year_a == pairs.birth_year_b).astype(int)
    pairs["is_match"] = (pairs.person_id_a == pairs.person_id_b).astype(int)
    return pairs


FEATURES = ["agree_first_name", "agree_last_name", "agree_street", "agree_postcode"]


def fs_weights(train: pd.DataFrame, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate m=P(agree|match), u=P(agree|nonmatch), with Laplace smoothing."""
    m_rows = train.loc[train.is_match == 1, FEATURES]
    u_rows = train.loc[train.is_match == 0, FEATURES]
    m = ((m_rows.sum(axis=0) + alpha) / (len(m_rows) + 2 * alpha)).to_numpy(float)
    u = ((u_rows.sum(axis=0) + alpha) / (len(u_rows) + 2 * alpha)).to_numpy(float)
    return m, u


def score(pairs: pd.DataFrame, m: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Fellegi-Sunter log-likelihood ratio, assuming conditional feature independence."""
    x = pairs[FEATURES].to_numpy(dtype=float)
    return (x * np.log(m / u) + (1 - x) * np.log((1 - m) / (1 - u))).sum(axis=1)


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def main():
    OUT.mkdir(exist_ok=True)
    a, b = generate_data()
    pairs = compare(candidates(a, b))

    # Split by PERSON, preventing the same real person's positive pair from leaking between sets.
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(a.person_id.to_numpy())
    n_train = int(.6 * len(ids)); n_val = int(.2 * len(ids))
    groups = {"train": set(ids[:n_train]), "validation": set(ids[n_train:n_train + n_val]), "test": set(ids[n_train + n_val:])}
    subsets = {k: pairs.loc[pairs.person_id_a.isin(v)].copy() for k, v in groups.items()}
    m, u = fs_weights(subsets["train"])
    for part in subsets.values():
        part["score"] = score(part, m, u)
    val = subsets["validation"]
    threshold_options = np.r_[np.nextafter(val.score.min(), -np.inf), np.unique(val.score), np.nextafter(val.score.max(), np.inf)]
    threshold = max(threshold_options, key=lambda t: (metrics(val.is_match, val.score >= t)["f1"], t))
    test = subsets["test"]
    test["predicted_match"] = test.score >= threshold
    result = metrics(test.is_match, test.predicted_match)

    # All actual matches must be accounted for, even if blocking missed them.
    actual_test = b.loc[b.person_id.isin(groups["test"]) & b.person_id.isin(set(a.person_id))].shape[0]
    true_positive = int(((test.is_match == 1) & test.predicted_match).sum())
    result["end_to_end_recall"] = true_positive / actual_test if actual_test else 0.0
    result["blocking_recall"] = int(test.is_match.sum()) / actual_test if actual_test else 0.0
    a.drop(columns="person_id").to_csv(OUT / "register_a.csv", index=False)
    b.drop(columns="person_id").to_csv(OUT / "register_b.csv", index=False)
    test[["record_id_a", "record_id_b", "score", "predicted_match", "is_match"]].sort_values("score", ascending=False).to_csv(OUT / "test_pairs.csv", index=False)
    
    print(f"Records A={len(a)}, B={len(b)}; candidate pairs={len(pairs)}")
    print("m probabilities:", dict(zip(FEATURES, np.round(m, 3))))
    print("u probabilities:", dict(zip(FEATURES, np.round(u, 3))))
    print(f"Selected threshold (validation F1): {threshold:.3f}")
    print("Held-out test results:")
    for k, v in result.items():
        print(f"  {k}: {v:.3f}")
    print(f"CSV outputs: {OUT}")


if __name__ == "__main__":
    main()

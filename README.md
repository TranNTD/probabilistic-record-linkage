# Probabilistic Record Linkage – Mini Project

An end-to-end demonstration of **Fellegi–Sunter probabilistic record linkage**, relevant to deduplicating and linking synthetic administrative registers. Uses *only synthetic data*.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Pipeline

1. Generate synthetic person records for two registers, including spelling differences and an unmatched population.
2. Generate candidate record pairs using **birth-year blocking** (efficient alternative to a Cartesian product).
3. Compare names and addresses using fuzzy string similarity, plus exact postal code matching.
4. Estimate Fellegi–Sunter **m** (agreement given a true match) and **u** (agreement given a non-match) probabilities from labeled synthetic training pairs with Laplace smoothing.
5. Calculate the record-pair log likelihood ratio:

   `weight = Σ [agree × log(m/u) + disagree × log((1-m)/(1-u))]`

6. Choose a decision threshold to maximize F1 on a separate validation partition.
7. Evaluate precision, candidate-level recall, F1, **blocking recall**, and **end-to-end recall** on held-out persons.

## Outputs

Running the script creates `outputs/register_a.csv`, `outputs/register_b.csv` and `outputs/test_pairs.csv`. The `person_id` ground truth is **never exported** with raw registers. It is retained in the test scoring output for evaluation only.

## Key methodological caveats

- This is a **supervised estimation of m/u probabilities using synthetic ground truth**; real-world linkage may lack labeled matches.
- The conditional independence assumption between comparison features is simplified; for example, name and address similarities may be dependent.
- Blocking on birth year assumes it is error-free and misses matches when that year is inaccurate.
- The model classifies **pairs** and does not enforce one-to-one matching, global consistency, or manual review of ambiguous pairs.
- Avoid interpreting the numeric score as a calibrated posterior probability. It is a *log likelihood ratio* without a prior.
- Processing sensitive administrative records requires data protection, pseudonymization, secure infrastructure, audit trails, and institutional authorization.


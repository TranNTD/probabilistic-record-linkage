import pandas as pd

# Read the model results
pairs = pd.read_csv("outputs/test_pairs.csv")

# Read the complete incoming register B
register_b = pd.read_csv("outputs/register_b.csv")

# Step 1: Select matched pairs
matches = pairs[pairs["predicted_match"] == True].copy()

# Step 2: Keep the highest-scoring match per B record
matches = (
    matches.sort_values("score", ascending=False)
    .drop_duplicates(subset="record_id_b")
)

# Step 3: Identify matched B records
matched_ids = set(matches["record_id_b"])

# Step 4: Separate new B records
new_records = register_b[
    ~register_b["record_id"].isin(matched_ids)
].copy()

# Step 5: Save results
matches.to_csv("outputs/matched_records.csv", index=False)
new_records.to_csv("outputs/new_records.csv", index=False)

print("Matched records:", len(matches))
print("New records:", len(new_records))
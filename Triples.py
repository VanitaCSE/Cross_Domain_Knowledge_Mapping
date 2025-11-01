import pandas as pd
import ast

# Load your dataset
data = pd.read_csv("relations_extracted_dataset.csv")

# Create an empty list for triples
triples_list = []

# Loop through each row and extract (subject, relation, object)
for _, row in data.iterrows():
    # Convert string to Python object safely
    rels = ast.literal_eval(row["relations"])
    for r in rels:
        # Each r looks like ('selection', 'relate', 'Sociology')
        subject, relation, obj = r
        triples_list.append({
            "subject": subject,
            "relation": relation,
            "object": obj
        })

# Convert list to DataFrame
triples_df = pd.DataFrame(triples_list)

# Save as triples.csv
triples_df.to_csv("triples.csv", index=False)

print("✅ triples.csv created successfully!")
print(triples_df.head())

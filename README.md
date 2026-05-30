# DANDE-DAYAKAR

## Environmental Toxicity Prediction

Yes — this project can now **build a starter compound set** and run a baseline toxicity model.

### Added compound builder
Use `build_example_compounds()` to create a small in-code dataset with common environmental contaminants.

Compounds included:
- Benzene
- Toluene
- Atrazine
- Phenol
- Bisphenol A
- Chloroform

### Quick start
```python
from environmental_toxicity_prediction import (
    build_example_compounds,
    train_toxicity_model,
)

# 1) Build starter compounds
df = build_example_compounds()

# 2) Train baseline model
result = train_toxicity_model(
    data=df,
    target_col="toxicity_index",
    categorical_cols=["compound", "family"],
)

print(df)
print(result)
```

### Notes
- `toxicity_index` values are illustrative so you can test the workflow quickly.
- For production use, replace with measured endpoints (LC50/EC50/NOEC), plus richer descriptors.

# Environmental Toxicity Prediction Report

## Scope
This report summarizes the built-in starter compound set for the environmental toxicity prediction workflow.

## Compound Dataset
| Compound | Family | logKow | Solubility (mg/L) | Molecular Weight | pH | Temperature (°C) | Toxicity Index |
|---|---:|---:|---:|---:|---:|---:|---:|
| Benzene | VOC | 2.13 | 1790.0 | 78.11 | 7.0 | 20.0 | 62.0 |
| Toluene | VOC | 2.73 | 530.0 | 92.14 | 7.0 | 20.0 | 58.0 |
| Atrazine | Pesticide | 2.61 | 33.0 | 215.68 | 7.0 | 20.0 | 71.0 |
| Phenol | Aromatic | 1.46 | 84000.0 | 94.11 | 7.0 | 20.0 | 49.0 |
| Bisphenol A | Endocrine disruptor | 3.64 | 300.0 | 228.29 | 7.0 | 20.0 | 76.0 |
| Chloroform | Halogenated solvent | 1.97 | 7950.0 | 119.38 | 7.0 | 20.0 | 66.0 |

## Summary Metrics
- Number of compounds: 6
- Average illustrative toxicity index: 63.67
- Lowest illustrative toxicity index: Phenol (49.0)
- Highest illustrative toxicity index: Bisphenol A (76.0)

## Modeling Notes
- The toxicity index values are illustrative placeholders for demo use.
- Replace `toxicity_index` with measured LC50, EC50, NOEC, or a similar endpoint before making scientific decisions.
- The training pipeline accepts the generated compounds directly when pandas and scikit-learn are installed.


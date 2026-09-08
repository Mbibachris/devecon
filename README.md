# devecon

[![tests](https://github.com/Mbibachris/devecon/actions/workflows/ci.yml/badge.svg)](https://github.com/Mbibachris/devecon/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A development-economics toolkit for Python. Clean, tested, survey-weight-aware
implementations of the measures applied economists actually use — starting with
poverty, with inequality and decomposition methods planned.

Most academic code for these measures is a loose script with no tests. `devecon`
is built as a proper package: every measure is checked against known results, the
economic identities are enforced as tests, and the whole suite runs on every push
across Python 3.9–3.12.

## What's implemented today

| Area | Functions | Notes |
|------|-----------|-------|
| **Unidimensional poverty** | `fgt` | Foster–Greer–Thorbecke: headcount, gap, severity (α = 0, 1, 2), with survey weights |
| **Multidimensional poverty** | `alkire_foster` | Adjusted headcount (M₀ = H·A), dual-cutoff counting, indicator + sample weights |
| **Identification** | `deprive`, `deprive_in` | Turn raw achievements into deprivation flags; thresholds, ordinal/Likert, and categorical indicators |
| **Assembly** | `build_deprivation_matrix`, `build_nested_weights` | Build an aligned deprivation matrix and matching weight vector from a single spec |

## Install

Until the first PyPI release, install from source:

```bash
git clone https://github.com/Mbibachris/devecon.git
cd devecon
pip install -e ".[pandas]"
```

The core measures depend only on NumPy. The DataFrame-based assembly helpers
(`build_deprivation_matrix`) additionally need pandas, hence the `[pandas]` extra.

## Quickstart

The FGT poverty index — headcount ratio (α = 0), poverty gap (α = 1), and
squared gap / severity (α = 2):

```python
from devecon import fgt

fgt([10, 20, 30, 40], poverty_line=25, alpha=0)   # 0.5  — half are poor
fgt([10, 20, 30, 40], poverty_line=25, alpha=1)   # 0.2  — average normalized gap
```

Survey weights are supported everywhere:

```python
fgt([10, 40], poverty_line=25, alpha=0, weights=[2, 1])   # weighted headcount
```

## The multidimensional workflow

The real value is the full Alkire–Foster pipeline: from a raw survey DataFrame
to the multidimensional poverty numbers, with a single spec driving both the
deprivation matrix and the weights so their columns can never fall out of
alignment.

```python
import pandas as pd
from devecon import (
    build_nested_weights,
    build_deprivation_matrix,
    alkire_foster,
)

# 1. Raw survey achievements
df = pd.DataFrame({
    "school_years": [3, 6, 9, 12],
    "water_src":    ["surface", "surface", "piped", "piped"],
})

# 2. Dimension structure -> aligned indicator weights
structure = {
    "education": {"weight": 1/2, "indicators": ["school"]},
    "living":    {"weight": 1/2, "indicators": ["water"]},
}
names, weights = build_nested_weights(structure)

# 3. Identification rules -> aligned 0/1 deprivation matrix
spec = [
    {"name": "school", "column": "school_years", "op": "<", "cutoff": 6},
    {"name": "water",  "column": "water_src", "deprived_categories": {"surface"}},
]
mat_names, matrix, kept = build_deprivation_matrix(df, spec)

# 4. Aggregate into H (incidence), A (intensity), M0 (adjusted headcount)
result = alkire_foster(matrix, cutoff_k=0.5, indicator_weights=weights)
print(result)
# AFResult(H=0.5, A=0.75, M0=0.375)
```

`H` is the share of people who are multidimensionally poor, `A` the average
breadth of deprivation among the poor, and `M0 = H × A` the adjusted headcount
ratio. That identity holds by construction — and is enforced as a test.

## Design notes

A few choices worth surfacing, because they reflect how the measures are meant
to be used on real data:

- **Survey weights throughout.** Real welfare data (LSMS-ISA, DHS, GLSS,
  Afrobarometer) ships with sampling weights. Every measure accepts them, and
  defaults to equal weights when omitted.

- **Identification is separated from aggregation.** Deciding *who is deprived*
  in each indicator (`deprive`, `deprive_in`) is kept distinct from combining
  those into a poverty measure (`alkire_foster`) — mirroring the structure of
  the Alkire–Foster method itself. Ordinal/Likert indicators are handled by an
  explicit deprivation cutoff, the standard treatment for ordinal data.

- **The deprivation cutoff is stated as an operator.** Writing
  `deprive(years, "<", 6)` makes the rule unambiguous: no hidden assumption
  about whether low or high is deprived, or whether the boundary itself counts.

- **Missing data is handled honestly.** An unknown achievement yields an unknown
  deprivation (`NaN`), never a silent "not deprived." Assembly then requires an
  explicit policy — raise, drop the row, or treat as non-deprived — so poverty
  is never understated by accident.

- **Weights can't silently misalign.** A single spec drives both the matrix
  column order and the weight order, and nested weights preserve each
  dimension's share — preventing the common bug where equal-per-indicator
  weighting quietly reweights the dimensions.

## Status and roadmap

**Implemented:** unidimensional (FGT) and multidimensional (Alkire–Foster)
poverty, with the identification and assembly layers that feed them.

**Planned:**

- **Inequality** — Gini, Generalized Entropy (Theil-T, Theil-L), Atkinson,
  Lorenz curves
- **Decomposition** — Oaxaca–Blinder (two-fold and three-fold)
- **Weighting options** — entropy and PCA-based indicator weights alongside the
  nested scheme
- **Poverty extensions** — subgroup and dimensional decomposition, cardinal
  measures (M₁, M₂)
- An R port sharing the same test-verified results

## Development

```bash
pip install -e ".[dev]"   # installs pytest, pandas, and coverage tooling
pytest                    # run the full suite
```

The test suite covers each measure against known results, the M₀ = H·A
identity, the boundary behaviour of deprivation cutoffs, weight-normalization
invariants, and each missing-data policy. CI runs it on Python 3.9, 3.10, 3.11,
and 3.12 on every push.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use `devecon` in research, please cite it. A `CITATION.cff` file will be
added with the first tagged release.

## Author

**Christopher Mbiba** — MPhil Economics, University of Cape Coast.
[GitHub](https://github.com/Mbibachris) · [ORCID](https://orcid.org/0009-0003-7114-719X)
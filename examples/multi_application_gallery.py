"""Examples gallery: one engine, three multidimensional indices.

The Alkire-Foster method is not a poverty method -- it is a general
counting-based multidimensional measurement method. Poverty is only its
most famous application. This script computes three completely different
indices using the *same* devecon functions (build_nested_weights,
build_deprivation_matrix, alkire_foster), changing nothing but the
dimensions, indicators, and cutoffs.

Run:  python examples/multi_application_gallery.py
"""

import pandas as pd

from devecon import (
    af_decompose,
    alkire_foster,
    build_deprivation_matrix,
    build_nested_weights,
)


def multidimensional_poverty():
    """Global-MPI style index: education, health, living standards."""
    df = pd.DataFrame({
        "years_schooling": [2, 8, 5, 11, 6, 1],
        "child_attends":   ["no", "yes", "yes", "yes", "no", "no"],
        "nutrition":       [0, 1, 0, 1, 1, 0],          # 1 = nourished
        "child_mortality": ["yes", "no", "no", "no", "yes", "yes"],
        "electricity":     ["no", "yes", "no", "yes", "yes", "no"],
        "water":           ["surface", "piped", "well", "piped",
                            "surface", "surface"],
    })
    structure = {
        "education": {"weight": 1/3, "indicators": ["schooling", "attendance"]},
        "health":   {"weight": 1/3, "indicators": ["nutrition", "mortality"]},
        "living":   {"weight": 1/3, "indicators": ["electricity", "water"]},
    }
    spec = [
        {"name": "schooling",   "column": "years_schooling",  "op": "<", "cutoff": 6},
        {"name": "attendance",  "column": "child_attends",    "deprived_categories": {"no"}},
        {"name": "nutrition",   "column": "nutrition",        "op": "<", "cutoff": 1},
        {"name": "mortality",   "column": "child_mortality",  "deprived_categories": {"yes"}},
        {"name": "electricity", "column": "electricity",      "deprived_categories": {"no"}},
        {"name": "water",       "column": "water",            "deprived_categories": {"surface", "well"}},
    ]
    return _run(df, structure, spec, "Multidimensional Poverty Index (MPI)")


def energy_poverty():
    """Multidimensional Energy Poverty Index: access and services."""
    df = pd.DataFrame({
        "electricity_access": ["no", "yes", "no", "yes", "no"],
        "cooking_fuel":       ["wood", "gas", "charcoal", "gas", "wood"],
        "lighting":           ["kerosene", "electric", "kerosene", "electric", "candle"],
        "appliances_owned":   [0, 3, 1, 4, 0],
        "phone_access":       ["no", "yes", "no", "yes", "no"],
    })
    structure = {
        "access":   {"weight": 1/2, "indicators": ["electricity", "cooking"]},
        "services": {"weight": 1/2, "indicators": ["lighting", "appliances", "phone"]},
    }
    spec = [
        {"name": "electricity", "column": "electricity_access", "deprived_categories": {"no"}},
        {"name": "cooking",     "column": "cooking_fuel",       "deprived_categories": {"wood", "charcoal"}},
        {"name": "lighting",    "column": "lighting",           "deprived_categories": {"kerosene", "candle"}},
        {"name": "appliances",  "column": "appliances_owned",   "op": "<", "cutoff": 2},
        {"name": "phone",       "column": "phone_access",       "deprived_categories": {"no"}},
    ]
    return _run(df, structure, spec, "Multidimensional Energy Poverty Index (MEPI)")


def child_wellbeing():
    """MODA-style multidimensional child deprivation: health, education, WASH."""
    df = pd.DataFrame({
        "immunized":            ["no", "yes", "yes", "no", "yes"],
        "stunted":              ["yes", "no", "no", "yes", "no"],
        "in_school":            ["no", "yes", "no", "yes", "yes"],
        "school_years_behind":  [3, 0, 2, 0, 1],
        "safe_water":           ["no", "yes", "no", "yes", "yes"],
        "sanitation":           ["no", "yes", "no", "no", "yes"],
    })
    structure = {
        "health":    {"weight": 1/3, "indicators": ["immunization", "nutrition"]},
        "education": {"weight": 1/3, "indicators": ["enrollment", "progression"]},
        "wash":      {"weight": 1/3, "indicators": ["water", "sanitation"]},
    }
    spec = [
        {"name": "immunization", "column": "immunized",           "deprived_categories": {"no"}},
        {"name": "nutrition",    "column": "stunted",             "deprived_categories": {"yes"}},
        {"name": "enrollment",   "column": "in_school",           "deprived_categories": {"no"}},
        {"name": "progression",  "column": "school_years_behind", "op": ">=", "cutoff": 2},
        {"name": "water",        "column": "safe_water",          "deprived_categories": {"no"}},
        {"name": "sanitation",   "column": "sanitation",          "deprived_categories": {"no"}},
    ]
    return _run(df, structure, spec, "Multidimensional Child Wellbeing (MODA-style)",
                show_drivers=True)


def _run(df, structure, spec, title, show_drivers=False):
    """Shared plumbing: the identical devecon pipeline for every index."""
    _, weights = build_nested_weights(structure)
    mat_names, matrix, _ = build_deprivation_matrix(df, spec)
    res = alkire_foster(matrix, cutoff_k=1/3, indicator_weights=weights)

    print("=" * 68)
    print(title)
    print("=" * 68)
    print(f"  H  (incidence) = {res.H:.3f}")
    print(f"  A  (intensity) = {res.A:.3f}")
    print(f"  M0 (adj. head) = {res.M0:.3f}")

    if show_drivers:
        dec = af_decompose(matrix, cutoff_k=1/3, indicator_weights=weights,
                           indicator_names=mat_names)
        ranked = sorted(zip(dec.indicator_names, dec.contributions),
                        key=lambda t: -t[1])
        print("  Top drivers:")
        for name, contrib in ranked[:3]:
            print(f"     {name}: {contrib * 100:.1f}% of M0")
    print()
    return res


if __name__ == "__main__":
    multidimensional_poverty()
    energy_poverty()
    child_wellbeing()
    print("Same alkire_foster + build_* engine, three different indices.")
    print("The method measures joint deprivation across dimensions --")
    print("what those dimensions *are* is entirely up to you.")
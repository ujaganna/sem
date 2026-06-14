from itertools import combinations, chain
from typing import Dict, List


def powerset(lst):
    return chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1))


def generate_permutations(
    exogenous: List[str],
    endogenous: List[str],
    mediators: List[str],
    max_models: int = 60,
) -> List[Dict]:
    """
    Enumerate SEM structural permutations by varying:
    - Which subset of mediators to include (powerset)
    - Whether a direct IV→DV path exists (partial vs full mediation)

    Each permutation dict has keys:
        id, mediators_used, mediation_type, include_direct, description
    """
    models = []

    for med_subset in powerset(mediators):
        med_list = list(med_subset)

        for include_direct in ([True, False] if med_list else [True]):
            if not med_list and not include_direct:
                continue  # nothing to estimate

            med_type = (
                "No Mediation" if not med_list
                else ("Partial" if include_direct else "Full")
            )

            iv_str = " + ".join(exogenous) if exogenous else "—"
            med_str = " + ".join(med_list) if med_list else "None"
            dv_str = " + ".join(endogenous) if endogenous else "—"
            direct_str = "Yes" if include_direct else "No"

            description = (
                f"{iv_str} → [{med_str}] → {dv_str}  |  Direct: {direct_str}"
            )

            models.append({
                "id": f"M{len(models) + 1:02d}",
                "mediators_used": med_list,
                "mediation_type": med_type,
                "include_direct": include_direct,
                "description": description,
            })

            if len(models) >= max_models:
                return models

    return models

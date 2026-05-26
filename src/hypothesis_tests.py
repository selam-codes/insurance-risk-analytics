"""
Hypothesis Testing Utilities for Insurance Risk Analytics

Reusable functions for A/B hypothesis testing on insurance data.
Supports chi-squared tests (categorical KPIs) and t-test / z-test
(numerical KPIs).

KPIs:
    - Claim Frequency: proportion of policies with at least one claim.
    - Claim Severity:  average claim amount, given a claim occurred.
    - Margin:          TotalPremium − TotalClaims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def add_claim_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a boolean ``HasClaim`` column."""
    out = df.copy()
    out["HasClaim"] = (out["TotalClaims"] > 0).astype(int)
    return out


def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``Margin`` column (Premium − Claims)."""
    out = df.copy()
    out["Margin"] = out["TotalPremium"] - out["TotalClaims"]
    return out


def claim_severity(df: pd.DataFrame) -> pd.Series:
    """Return TotalClaims for rows where a claim occurred."""
    return df.loc[df["TotalClaims"] > 0, "TotalClaims"]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def chi_squared_test(
    group_a: pd.Series,
    group_b: pd.Series,
) -> tuple[float, float]:
    """
    Chi-squared test for comparing proportions of a binary variable
    (e.g. HasClaim) between two groups.

    Parameters
    ----------
    group_a, group_b : pd.Series
        Binary (0/1) series for each group.

    Returns
    -------
    chi2 : float
        The chi-squared statistic.
    p_value : float
        Two-sided p-value.
    """
    # Build a 2×2 contingency table
    count_a = group_a.value_counts().reindex([0, 1], fill_value=0)
    count_b = group_b.value_counts().reindex([0, 1], fill_value=0)
    contingency = np.array([count_a.values, count_b.values])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return chi2, p_value


def chi_squared_test_multi(
    df: pd.DataFrame,
    group_col: str,
    kpi_col: str = "HasClaim",
) -> tuple[float, float]:
    """
    Chi-squared test across *all* categories of ``group_col``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``group_col`` and ``kpi_col`` columns.
    group_col : str
        The categorical column (e.g. 'Province').
    kpi_col : str
        Binary 0/1 column to compare across groups.

    Returns
    -------
    chi2, p_value
    """
    contingency = pd.crosstab(df[group_col], df[kpi_col])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return chi2, p_value


def two_sample_ttest(
    group_a: pd.Series,
    group_b: pd.Series,
    equal_var: bool = False,
) -> tuple[float, float]:
    """
    Welch's two-sample t-test (unequal variances by default).

    Returns
    -------
    t_stat, p_value
    """
    t_stat, p_value = stats.ttest_ind(
        group_a.dropna(), group_b.dropna(), equal_var=equal_var
    )
    return t_stat, p_value


def two_sample_ztest(
    group_a: pd.Series,
    group_b: pd.Series,
) -> tuple[float, float]:
    """
    Two-sample z-test for means of large samples (CLT applies).

    Returns
    -------
    z_stat, p_value
    """
    n_a, n_b = len(group_a.dropna()), len(group_b.dropna())
    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a, var_b = group_a.var(), group_b.var()

    se = np.sqrt(var_a / n_a + var_b / n_b)
    z_stat = (mean_a - mean_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


# ---------------------------------------------------------------------------
# Group balancing / selection
# ---------------------------------------------------------------------------

def select_two_groups(
    df: pd.DataFrame,
    feature_col: str,
    cat_a: str,
    cat_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into two groups based on two category values in *feature_col*.
    """
    grp_a = df[df[feature_col] == cat_a].copy()
    grp_b = df[df[feature_col] == cat_b].copy()
    return grp_a, grp_b


def compare_group_balance(
    grp_a: pd.DataFrame,
    grp_b: pd.DataFrame,
    balance_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Check whether *grp_a* and *grp_b* are statistically equivalent on
    the given ``balance_cols`` (using chi-squared for categorical and
    t-test for numerical columns).

    Returns a DataFrame with columns:
        column, test, statistic, p_value, equivalent
    """
    rows = []
    for col in balance_cols:
        if col not in grp_a.columns or col not in grp_b.columns:
            continue
        if not pd.api.types.is_numeric_dtype(grp_a[col]) or grp_a[col].nunique() < 10:
            # Categorical → chi-squared on value_counts
            ct = pd.crosstab(
                pd.concat([grp_a[[col]], grp_b[[col]]]).reset_index(drop=True)[col],
                pd.Series(
                    ["A"] * len(grp_a) + ["B"] * len(grp_b), name="group"
                ),
            )
            chi2, p, _, _ = stats.chi2_contingency(ct)
            rows.append(
                {"column": col, "test": "chi2", "statistic": chi2,
                 "p_value": p, "equivalent": p >= alpha}
            )
        else:
            # Numerical → t-test
            t, p = stats.ttest_ind(
                grp_a[col].dropna(), grp_b[col].dropna(), equal_var=False
            )
            rows.append(
                {"column": col, "test": "t-test", "statistic": t,
                 "p_value": p, "equivalent": p >= alpha}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HypothesisResult:
    """Container for a single hypothesis-test outcome."""

    hypothesis: str
    kpi: str
    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    group_a_label: str = ""
    group_b_label: str = ""
    group_a_metric: float | None = None
    group_b_metric: float | None = None
    interpretation: str = ""

    @property
    def reject(self) -> bool:
        return self.p_value < self.alpha

    @property
    def decision(self) -> str:
        return "Reject H₀" if self.reject else "Fail to reject H₀"

    def summary_dict(self) -> dict:
        return {
            "Hypothesis": self.hypothesis,
            "KPI": self.kpi,
            "Group A": self.group_a_label,
            "Group B": self.group_b_label,
            "Test": self.test_name,
            "Statistic": round(self.statistic, 4),
            "p-value": f"{self.p_value:.2e}" if self.p_value < 0.001 else round(self.p_value, 4),
            "α": self.alpha,
            "Decision": self.decision,
        }


def results_table(results: list[HypothesisResult]) -> pd.DataFrame:
    """Return a tidy summary DataFrame from a list of HypothesisResult."""
    return pd.DataFrame([r.summary_dict() for r in results])


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_frequency_test(
    df: pd.DataFrame,
    feature_col: str,
    cat_a: str,
    cat_b: str,
    hypothesis_label: str,
    alpha: float = 0.05,
) -> HypothesisResult:
    """
    Run a claim-frequency (chi-squared) test between two groups.
    """
    df = add_claim_flag(df)
    grp_a, grp_b = select_two_groups(df, feature_col, cat_a, cat_b)

    freq_a = grp_a["HasClaim"].mean()
    freq_b = grp_b["HasClaim"].mean()
    chi2, p = chi_squared_test(grp_a["HasClaim"], grp_b["HasClaim"])

    return HypothesisResult(
        hypothesis=hypothesis_label,
        kpi="Claim Frequency",
        test_name="Chi-squared",
        statistic=chi2,
        p_value=p,
        alpha=alpha,
        group_a_label=str(cat_a),
        group_b_label=str(cat_b),
        group_a_metric=freq_a,
        group_b_metric=freq_b,
    )


def run_severity_test(
    df: pd.DataFrame,
    feature_col: str,
    cat_a: str,
    cat_b: str,
    hypothesis_label: str,
    alpha: float = 0.05,
) -> HypothesisResult:
    """
    Run a claim-severity (t-test) comparison between two groups.
    Only considers rows where TotalClaims > 0.
    """
    grp_a, grp_b = select_two_groups(df, feature_col, cat_a, cat_b)
    sev_a = claim_severity(grp_a)
    sev_b = claim_severity(grp_b)

    if len(sev_a) < 2 or len(sev_b) < 2:
        return HypothesisResult(
            hypothesis=hypothesis_label,
            kpi="Claim Severity",
            test_name="t-test (insufficient data)",
            statistic=float("nan"),
            p_value=float("nan"),
            alpha=alpha,
            group_a_label=str(cat_a),
            group_b_label=str(cat_b),
            group_a_metric=sev_a.mean() if len(sev_a) else None,
            group_b_metric=sev_b.mean() if len(sev_b) else None,
        )

    t, p = two_sample_ttest(sev_a, sev_b)
    return HypothesisResult(
        hypothesis=hypothesis_label,
        kpi="Claim Severity",
        test_name="t-test",
        statistic=t,
        p_value=p,
        alpha=alpha,
        group_a_label=str(cat_a),
        group_b_label=str(cat_b),
        group_a_metric=sev_a.mean(),
        group_b_metric=sev_b.mean(),
    )


def run_margin_test(
    df: pd.DataFrame,
    feature_col: str,
    cat_a: str,
    cat_b: str,
    hypothesis_label: str,
    alpha: float = 0.05,
    use_ztest: bool = True,
) -> HypothesisResult:
    """
    Run a margin (TotalPremium − TotalClaims) test between two groups.
    Uses z-test by default (large samples); falls back to t-test.
    """
    df = add_margin(df)
    grp_a, grp_b = select_two_groups(df, feature_col, cat_a, cat_b)

    margin_a = grp_a["Margin"]
    margin_b = grp_b["Margin"]

    if use_ztest and len(margin_a) > 30 and len(margin_b) > 30:
        stat, p = two_sample_ztest(margin_a, margin_b)
        test_name = "z-test"
    else:
        stat, p = two_sample_ttest(margin_a, margin_b)
        test_name = "t-test"

    return HypothesisResult(
        hypothesis=hypothesis_label,
        kpi="Margin",
        test_name=test_name,
        statistic=stat,
        p_value=p,
        alpha=alpha,
        group_a_label=str(cat_a),
        group_b_label=str(cat_b),
        group_a_metric=margin_a.mean(),
        group_b_metric=margin_b.mean(),
    )

import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ==============================
# Tax & salary helper functions
# ==============================

def compute_progressive_tax(gain: float) -> float:
    """
    Compute Spanish capital gains tax using progressive brackets.

    Parameters
    ----------
    gain : float
        Total capital gain to be taxed.

    Returns
    -------
    float
        Total tax due on the gain.
    """
    if gain <= 0:
        return 0.0

    remaining = gain
    tax = 0.0

    # Brackets (approximate, combined state + regional)
    limits = [6000, 50000, 200000]  # cumulative
    rates = [0.19, 0.21, 0.23]
    prev = 0.0

    for limit, rate in zip(limits, rates):
        if remaining <= 0:
            break
        taxable = min(remaining, limit - prev)
        if taxable > 0:
            tax += taxable * rate
            remaining -= taxable
            prev = limit

    # Final bracket
    if remaining > 0:
        tax += remaining * 0.26

    return tax


def compute_salary_net(gross_annual: float):
    """
    Very rough approximation of NET salary from gross annual salary in Spain.

    - Applies ~6.35% Social Security contribution up to a capped base.
    - Applies approximate progressive IRPF rates on the remaining base.
    - Does NOT take into account personal / family allowances or specific deductions.

    Returns
    -------
    net_annual : float
    ss_contrib : float
    irpf : float
    effective_total_rate : float
        Total effective tax+SS rate (0–1).
    """
    if gross_annual <= 0:
        return 0.0, 0.0, 0.0, 0.0

    # 1) Employee's Social Security (~6.35% up to a max base)
    ss_rate = 0.0635
    SS_MAX_ANNUAL_BASE = 60000.0  # rough approximation

    ss_base = min(gross_annual, SS_MAX_ANNUAL_BASE)
    ss_contrib = ss_base * ss_rate

    # 2) IRPF base (simplified: gross - SS)
    base_irpf = max(0.0, gross_annual - ss_contrib)

    # Approx progressive IRPF brackets (can vary by region)
    limits = [12450, 20200, 35200, 60000, 300000]
    rates = [0.19, 0.24, 0.30, 0.37, 0.45]
    remaining = base_irpf
    prev = 0.0
    irpf = 0.0

    for limit, rate in zip(limits, rates):
        if remaining <= 0:
            break
        tramo = min(remaining, limit - prev)
        if tramo > 0:
            irpf += tramo * rate
            remaining -= tramo
            prev = limit

    if remaining > 0:
        irpf += remaining * 0.47  # > 300k

    net_annual = gross_annual - ss_contrib - irpf
    effective_total_rate = 1.0 - (net_annual / gross_annual) if gross_annual > 0 else 0.0

    return net_annual, ss_contrib, irpf, effective_total_rate


# ==============================
# JSON helpers for plans & portfolios
# ==============================

PORTFOLIO_FILE = "cartera.json"       # single unnamed portfolio (backwards compatibility)
PLANS_FILE = "planes.json"            # long-term goal plans, housing plans, etc.
PORTFOLIOS_FILE = "carteras.json"     # named portfolios
CUSTOM_ASSETS_FILE = "activos_custom.json"  # user-defined assets


def load_plans() -> dict:
    """Load all saved plans from PLANS_FILE."""
    if os.path.exists(PLANS_FILE):
        try:
            with open(PLANS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def save_plans(plans: dict) -> None:
    """Persist all plans to PLANS_FILE as JSON."""
    with open(PLANS_FILE, "w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)


def load_portfolios() -> dict:
    """
    Load named portfolios from PORTFOLIOS_FILE.

    Returns
    -------
    dict
        Mapping portfolio_name -> list of rows (dicts) representing the portfolio.
    """
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def save_portfolios(portfolios: dict) -> None:
    """Save named portfolios to PORTFOLIOS_FILE."""
    with open(PORTFOLIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolios, f, ensure_ascii=False, indent=2)


def load_custom_assets() -> list:
    """
    Load user-defined custom assets from CUSTOM_ASSETS_FILE.

    The file should contain a JSON list of objects with at least:
    - 'nombre'
    and optionally:
    - 'tipo', 'ticker', 'isin'
    """
    if os.path.exists(CUSTOM_ASSETS_FILE):
        try:
            with open(CUSTOM_ASSETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def save_custom_assets(custom_assets: list) -> None:
    """Save user-defined custom assets to CUSTOM_ASSETS_FILE."""
    with open(CUSTOM_ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(custom_assets, f, ensure_ascii=False, indent=2)


# ==============================
# Portfolio model & rebalancing logic
# ==============================

class Portfolio:
    """
    Simple portfolio representation used for rebalancing logic.
    """

    def __init__(
        self,
        holdings: dict,
        targets: dict,
        asset_types: dict | None = None,
    ):
        """
        Parameters
        ----------
        holdings : dict
            Mapping {asset_name: current_value_in_eur}.
        targets : dict
            Mapping {asset_name: target_weight (0–1)}. Should sum to ~1.
        asset_types : dict, optional
            Mapping {asset_name: type}, only used for display.
        """
        self.holdings = dict(holdings)
        self.targets = dict(targets)
        self.asset_types = dict(asset_types or {})

    def total_value(self) -> float:
        """Total current market value of the portfolio."""
        return float(sum(self.holdings.values()))

    def current_weights(self) -> dict:
        """
        Current weights (0–1) of each asset.

        Returns
        -------
        dict
            Mapping {asset_name: weight}.
        """
        total = self.total_value()
        if total <= 0:
            return {a: 0.0 for a in self.holdings}
        return {a: float(v) / total for a, v in self.holdings.items()}


def compute_contribution_plan(
    portfolio: Portfolio,
    monthly_contribution: float,
    rebalance_threshold: float = 0.0,
) -> dict:
    """
    Compute an ideal monthly contribution per asset to move towards target weights.

    Rules:
    - We never sell inside this function (only non-negative contributions).
    - If the ideal math solution would require selling, we cap that asset at 0 contribution.
    - Remaining money is redistributed among underweight assets.

    Note
    ----
    rebalance_threshold is kept in the signature for compatibility,
    but the threshold is applied later when considering optional sales.

    Returns
    -------
    dict
        Mapping {asset_name: contribution_in_eur}.
    """
    holdings = portfolio.holdings
    targets = portfolio.targets
    C = float(monthly_contribution)

    if C <= 0 or not holdings:
        return {a: 0.0 for a in holdings}

    total0 = portfolio.total_value()
    if total0 < 0:
        total0 = 0.0
    total1 = total0 + C

    # 1) Ideal contribution to end exactly at target weights
    raw_contribs: dict[str, float] = {}
    for a, h in holdings.items():
        t = float(targets.get(a, 0.0))
        ideal_value = t * total1
        raw = ideal_value - float(h)
        raw_contribs[a] = max(0.0, raw)  # never sell

    sum_raw = sum(raw_contribs.values())

    # Edge case: everyone looks roughly “on target”
    if sum_raw <= 0:
        sum_targets = sum(targets.values())
        if sum_targets <= 0:
            n = len(holdings)
            if n == 0:
                return {}
            uniform = C / n
            return {a: uniform for a in holdings}

        return {
            a: C * (float(targets.get(a, 0.0)) / sum_targets)
            for a in holdings
        }

    # 2) If ideal buys exceed available contribution, scale proportionally
    if sum_raw >= C:
        scale = C / sum_raw
        contribs = {a: raw_contribs[a] * scale for a in holdings}
    else:
        # 3) If we still have money left, allocate extra according to target weights
        leftover = C - sum_raw
        contribs = raw_contribs.copy()
        sum_targets = sum(targets.values())

        if sum_targets <= 0:
            n = len(holdings)
            extra_each = leftover / n if n > 0 else 0.0
            for a in holdings:
                contribs[a] += extra_each
        else:
            for a in holdings:
                contribs[a] += leftover * (float(targets.get(a, 0.0)) / sum_targets)

    return contribs


# ==============================
# Simple accumulation simulators
# ==============================

def simulate_constant_plan(
    current_total: float,
    monthly_contribution: float,
    years: int,
    annual_return: float,
    extra_savings: float = 0.0,
):
    """
    Simulate a constant monthly contribution plan with constant annual return.

    Returns
    -------
    final_value : float
    series : list[float]
        Estimated portfolio value after each month.
    """
    months = int(years * 12)
    r_m = float(annual_return) / 12.0
    value = float(current_total) + float(extra_savings)
    series = []

    for _ in range(months):
        value *= (1.0 + r_m)
        value += float(monthly_contribution)
        series.append(value)

    return value, series


def required_constant_monthly_for_goal(
    current_total: float,
    objective_final: float,
    years: int,
    annual_return: float,
    extra_savings: float = 0.0,
    tax_rate: float = 0.0,
) -> int:
    """
    Compute the constant monthly contribution required to reach a target value.

    tax_rate is kept for compatibility but not used (taxes are handled elsewhere).
    """
    months = int(years * 12)
    r_m = float(annual_return) / 12.0
    pv = float(current_total) + float(extra_savings)

    if abs(r_m) < 1e-12:
        needed = objective_final - pv
        if needed <= 0:
            return 0
        return int(round(needed / max(months, 1)))

    factor = (1.0 + r_m) ** months
    fv_pv = pv * factor

    if fv_pv >= objective_final:
        return 0

    pmt = (objective_final - fv_pv) * r_m / (factor - 1.0)
    return int(round(max(0.0, pmt)))


def simulate_dca_ramp(
    initial_monthly: float,
    final_monthly: float,
    years: int,
    annual_return: float,
    initial_value: float = 0.0,
):
    """
    Simulate a DCA plan with linearly increasing monthly contributions.

    Contributions grow linearly from initial_monthly to final_monthly over the horizon.
    """
    months = int(years * 12)
    r_m = float(annual_return) / 12.0
    value = float(initial_value)
    series = []

    if months <= 0:
        return value, series

    for m in range(months):
        frac = m / (months - 1) if months > 1 else 1.0
        contrib = float(initial_monthly) + (float(final_monthly) - float(initial_monthly)) * frac
        value *= (1.0 + r_m)
        value += contrib
        series.append(value)

    return value, series


def required_growing_monthlies_for_goal(
    current_total: float,
    objective_final: float,
    years: int,
    annual_return: float,
    initial_monthly: float,
    extra_savings: float = 0.0,
    tax_rate: float = 0.0,
):
    """
    Find the final monthly contribution for a linearly growing plan that hits the target.

    Returns
    -------
    final_monthly : int
    []
        Second return value kept only for backward compatibility.
    """
    initial_value = float(current_total) + float(extra_savings)

    # Check if starting monthly is already enough
    val0, _ = simulate_dca_ramp(
        initial_monthly=initial_monthly,
        final_monthly=initial_monthly,
        years=years,
        annual_return=annual_return,
        initial_value=initial_value,
    )
    if val0 >= objective_final:
        return int(round(initial_monthly)), []

    months = int(years * 12)
    low = float(initial_monthly)
    high = max(float(initial_monthly) * 3.0, 5000.0)

    for _ in range(40):
        mid = (low + high) / 2.0
        val_mid, _ = simulate_dca_ramp(
            initial_monthly=initial_monthly,
            final_monthly=mid,
            years=years,
            annual_return=annual_return,
            initial_value=initial_value,
        )
        if val_mid < objective_final:
            low = mid
        else:
            high = mid

    return int(round(high)), []


# ==============================
# Asset universe CSV loader
# ==============================

@st.cache_data
def load_universe_csv() -> pd.DataFrame:
    """
    Load the full instrument universe from the generated CSV (e.g. 'asset_universe.csv').

    Expected columns:
    ISIN, Name, Type, Region, Country, Country_Code, ETF_Provider,
    ETF_Subtype, Distribution, Currency_Name, Is_ADR, Page, Search_Key
    """
    try:
        df = pd.read_csv("asset_universe.csv")
        for col in ["ISIN", "Name", "Search_Key"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        for col in ["Type", "Region", "Country", "ETF_Provider", "ETF_Subtype", "Currency_Name"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame()


# ==============================
# Streamlit app
# ==============================

st.set_page_config(
    page_title="Portfolio Planner",
    page_icon="💶",
    layout="wide",
)

st.title("Portfolio Planner - Marcos Ibáñez")

st.markdown(
    """
This application helps you plan and manage your investments:

- **Portfolio rebalancing**: distribute your monthly contribution across assets to maintain target weights.  
- **Long-term goal planning**: estimate how much you need to invest (constant or growing contributions) to reach a target wealth.  
- **Housing plan**: plan a home down payment, extra costs, and mortgage.  
- **Portfolio analysis**: additional analytics and summaries.

Use the tabs below to navigate each module.
"""
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔁 Portfolio rebalancing",
        "🎯 Long-term goal",
        "🏠 Housing plan",
        "📊 Portfolio analysis",
    ]
)

# ============================
# TAB 1: MONTHLY REBALANCING
# ============================
with tab1:
    st.header("Monthly contribution & portfolio rebalancing")

    st.markdown(
        "1. Fill the table with your assets, type, current value and target weight.\n"
        "2. Enter how much you want to invest next month and the rebalance threshold.\n"
        "3. Review the suggested plan and the before/after weights."
    )

    custom_assets = load_custom_assets()

    def normalize_asset_type(raw_type: str) -> str:
        """
            Normalize raw asset type into the categories used in the app.
        """
        if not raw_type:
            return ""
        s = str(raw_type).strip().lower()
        if any(x in s for x in ["etf", "index fund", "fund", "fonds"]):
            return "ETF"
        if any(x in s for x in ["stock", "share", "equity", "aktion", "acción", "acciones"]):
            return "Stock"
        if any(x in s for x in ["bond", "renta fija", "obligat"]):
            return "Bond"
        if any(x in s for x in ["crypto", "bitcoin", "btc", "eth"]):
            return "Crypto"
        if any(x in s for x in ["derivative", "option", "future", "warrant"]):
            return "Derivative"
        if any(x in s for x in ["fund", "sicav", "fond"]):
            return "Fund"
        return "Other"

    @st.cache_data
    def build_universe_catalog() -> pd.DataFrame:
        """
        Build a smaller catalog (Name, ISIN, Type) from the full universe CSV.

        This is cached so we only process the large CSV once.
        """
        df = load_universe_csv()
        if df.empty:
            return pd.DataFrame(columns=["Name", "ISIN", "Type"])

        universe_small = df[["Name", "ISIN", "Type"]].copy()
        universe_small["Name"] = universe_small["Name"].astype(str).str.strip()
        universe_small["ISIN"] = universe_small["ISIN"].astype(str).str.strip().str.upper()
        universe_small["Type"] = universe_small["Type"].apply(normalize_asset_type)
        universe_small = universe_small.rename(columns={"Name": "Asset", "Type": "Type"})

        universe_small = universe_small[
            (universe_small["Asset"] != "") & (universe_small["ISIN"] != "")
        ]
        universe_small = universe_small.drop_duplicates(subset="ISIN").reset_index(drop=True)
        return universe_small[["Asset", "ISIN", "Type"]]

    # --- Custom assets UI ---
    with st.expander("➕ Add custom asset to your personal list"):
        custom_name = st.text_input(
            "Custom asset name",
            key="custom_asset_name",
        )
        custom_type = st.selectbox(
            "Custom asset type",
            options=["ETF", "Stock", "Bond", "Derivative", "Crypto", "Fund", "Other"],
            key="custom_asset_type",
        )
        custom_ticker = st.text_input(
            "Ticker (optional)",
            key="custom_asset_ticker",
        )
        custom_isin = st.text_input(
            "ISIN (optional)",
            key="custom_asset_isin",
        )

        if st.button("Add custom asset", key="btn_add_custom"):
            if not custom_name.strip():
                st.error("Asset name cannot be empty.")
            else:
                existing = load_custom_assets()
                existing.append(
                    {
                        "nombre": custom_name.strip(),
                        "tipo": custom_type,
                        "ticker": custom_ticker.strip(),
                        "isin": custom_isin.strip(),
                    }
                )
                save_custom_assets(existing)
                st.success(f"Custom asset '{custom_name}' added successfully.")
                st.rerun()

    # --- Portfolio table schema ---
    default_data = pd.DataFrame(
        {
            "Asset": pd.Series(dtype=str),
            "Type": pd.Series(dtype=str),
            "ISIN": pd.Series(dtype=str),
            "Current_value_€": pd.Series(dtype=float),
            "Target_weight_%": pd.Series(dtype=float),
        }
    )

    def ensure_portfolio_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the portfolio DataFrame has all required columns and types.
        """
        df = df.copy()
        for col, default in [
            ("Asset", ""),
            ("Type", ""),
            ("ISIN", ""),
            ("Current_value_€", 0.0),
            ("Target_weight_%", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

        df["Asset"] = df["Asset"].astype(str)
        df["Type"] = df["Type"].astype(str)
        df["ISIN"] = df["ISIN"].astype(str)
        df["Current_value_€"] = pd.to_numeric(df["Current_value_€"], errors="coerce").fillna(0.0)
        df["Target_weight_%"] = pd.to_numeric(df["Target_weight_%"], errors="coerce").fillna(0.0)

        return df[["Asset", "Type", "ISIN", "Current_value_€", "Target_weight_%"]]

    # Initialize portfolio from JSON if present
    if "portfolio_df" not in st.session_state:
        if os.path.exists(PORTFOLIO_FILE):
            try:
                loaded = pd.read_json(PORTFOLIO_FILE)
                st.session_state["portfolio_df"] = ensure_portfolio_schema(loaded)
            except Exception:
                st.session_state["portfolio_df"] = default_data.copy()
        else:
            st.session_state["portfolio_df"] = default_data.copy()

    # Build catalog from custom + universe
    custom_rows = [
        {
            "Asset": str(a.get("nombre", "")).strip(),
            "ISIN": str(a.get("isin", "")).strip().upper(),
            "Type": str(a.get("tipo", "")).strip(),
        }
        for a in custom_assets
    ]
    custom_catalog_df = pd.DataFrame(custom_rows) if custom_rows else pd.DataFrame(
        columns=["Asset", "ISIN", "Type"]
    )

    universe_small = build_universe_catalog()
    if not universe_small.empty:
        catalog_df = pd.concat([custom_catalog_df, universe_small], ignore_index=True)
    else:
        catalog_df = custom_catalog_df.copy()

    if not catalog_df.empty:
        catalog_df = catalog_df[(catalog_df["Asset"] != "") & (catalog_df["ISIN"] != "")]
        catalog_df = catalog_df.drop_duplicates(subset="ISIN").reset_index(drop=True)

    st.subheader("📋 Portfolio holdings")

    st.markdown("#### Add or update a holding")

    portfolio_df_current = ensure_portfolio_schema(st.session_state["portfolio_df"])

    # --- Asset selector ---
    if catalog_df.empty:
        st.info(
            "The full asset universe could not be loaded. "
            "You can add custom assets above and then select them here."
        )
        manual_name = st.text_input("Asset name")
        manual_isin = st.text_input("ISIN")
        manual_type = st.selectbox(
            "Asset type",
            options=["ETF", "Stock", "Bond", "Derivative", "Crypto", "Fund", "Other"],
        )
        selected_name = manual_name.strip()
        selected_isin = manual_isin.strip().upper()
        selected_type = manual_type
    else:
        catalog_for_select = catalog_df.copy()
        catalog_for_select["Label"] = (
            catalog_for_select["Asset"] + " (" + catalog_for_select["ISIN"] + ")"
        )

        placeholder_label = ""
        label_options = [placeholder_label] + catalog_for_select["Label"].tolist()

        if st.session_state.get("reset_asset_selector", False):
            st.session_state["asset_selector_label"] = placeholder_label
            st.session_state["reset_asset_selector"] = False

        selected_label = st.selectbox(
            "Search and select an asset",
            options=label_options,
            key="asset_selector_label",
        )

        if selected_label == placeholder_label:
            selected_isin = ""
            selected_name = ""
            selected_type = ""
        else:
            row_sel = catalog_for_select[catalog_for_select["Label"] == selected_label].iloc[0]
            selected_isin = row_sel["ISIN"]
            selected_name = row_sel["Asset"]
            selected_type = row_sel["Type"] or ""

    col_val, col_weight = st.columns(2)
    with col_val:
        value_sel = st.number_input(
            "Current value in portfolio (€)",
            min_value=0.0,
            step=50.0,
            value=0.0,
        )
    with col_weight:
        weight_sel = st.number_input(
            "Target weight (%) for this asset",
            min_value=0.0,
            step=1.0,
            value=0.0,
        )

    if st.button("➕ Add / update asset in portfolio"):
        if not selected_name and not selected_isin:
            st.error("You must provide at least a name or an ISIN for the asset.")
        else:
            df_cart = portfolio_df_current.copy()

            new_row = {
                "Asset": selected_name,
                "Type": selected_type,
                "ISIN": selected_isin,
                "Current_value_€": float(value_sel),
                "Target_weight_%": float(weight_sel),
            }

            if selected_isin:
                mask = df_cart["ISIN"].astype(str).str.upper().eq(selected_isin)
            else:
                mask = df_cart["Asset"].astype(str).str.strip().eq(selected_name)

            if mask.any():
                idx = df_cart.index[mask][0]
                for col, val in new_row.items():
                    if col in df_cart.columns:
                        df_cart.at[idx, col] = val
            else:
                df_cart = pd.concat([df_cart, pd.DataFrame([new_row])], ignore_index=True)

            st.session_state["portfolio_df"] = ensure_portfolio_schema(df_cart)
            st.success("Asset added/updated in portfolio.")
            st.session_state["reset_asset_selector"] = True
            st.rerun()

    # --- Portfolio table (read-only) ---
    df_holdings = ensure_portfolio_schema(st.session_state["portfolio_df"])
    if df_holdings.empty:
        st.info("You haven't added any assets to your portfolio yet.")
    else:
        st.markdown("#### Current portfolio")
        df_holdings_show = df_holdings.copy()
        if "Current_value_€" in df_holdings_show.columns:
            df_holdings_show["Current_value_€"] = df_holdings_show["Current_value_€"].round(2)
        if "Target_weight_%" in df_holdings_show.columns:
            df_holdings_show["Target_weight_%"] = df_holdings_show["Target_weight_%"].round(2)
        st.dataframe(df_holdings_show, use_container_width=True)

        # Delete assets by ISIN
        isins_in_portfolio = df_holdings["ISIN"].astype(str).str.strip().tolist()
        if any(isins_in_portfolio):
            unique_isins = sorted(set(i for i in isins_in_portfolio if i))
            isin_to_delete = st.multiselect(
                "Select assets to remove from the portfolio",
                options=unique_isins,
                format_func=lambda isin: (
                    df_holdings.loc[
                        df_holdings["ISIN"].astype(str).str.strip().eq(isin),
                        "Asset",
                    ].iloc[0] + f" ({isin})"
                ),
            )
            if isin_to_delete and st.button("🗑️ Remove selected"):
                mask_del = df_holdings["ISIN"].astype(str).str.strip().isin(isin_to_delete)
                df_holdings = df_holdings[~mask_del].reset_index(drop=True)
                st.session_state["portfolio_df"] = ensure_portfolio_schema(df_holdings)
                st.success("Assets removed from portfolio.")
                st.rerun()

    # --- Target weight sum & normalization ---
    show_normalize_button = False
    try:
        df_live = df_holdings.copy()
        df_live = df_live[df_live["Asset"].astype(str).str.strip().ne("")]
        sum_weights_live = float(df_live["Target_weight_%"].sum())
        st.markdown(
            f"**Sum of target weights (non-empty rows): {sum_weights_live:.2f}%**"
        )
        if not (98.5 <= sum_weights_live <= 101.5):
            show_normalize_button = True
    except Exception:
        show_normalize_button = False

    if show_normalize_button and st.button("⚖️ Normalize target weights to 100%", key="normalize_weights"):
        try:
            df_norm = df_holdings.copy()
            mask_valid = df_norm["Asset"].astype(str).str.strip().ne("")
            suma = df_norm.loc[mask_valid, "Target_weight_%"].sum()
            if suma > 0:
                serie_norm = df_norm.loc[mask_valid, "Target_weight_%"] / suma * 100.0
                df_norm.loc[mask_valid, "Target_weight_%"] = serie_norm.round(2)
                st.session_state["portfolio_df"] = df_norm
                st.success("Target weights normalized to 100% over non-empty rows.")
                st.rerun()
            else:
                st.error("The sum of target weights is 0. Cannot normalize.")
        except Exception as e:
            st.error(f"Could not normalize target weights: {e}")

    # Filter out empty rows for further calculations/plots
    df_holdings = df_holdings[df_holdings["Asset"].astype(str).str.strip().ne("")].copy()

    # --- Charts: current vs target allocation (more professional layout) ---
    if not df_holdings.empty:
        total_value = float(df_holdings["Current_value_€"].sum()) if "Current_value_€" in df_holdings else 0.0

        if total_value <= 0:
            st.info(
                "Enter a positive current value for your holdings to display the allocation charts."
            )
        else:
            # Compute current and target weights
            df_chart = df_holdings.copy()
            df_chart["Current_weight_%"] = (
                df_chart["Current_value_€"] / total_value * 100.0
            )
            df_chart["Target_weight_%"] = df_chart["Target_weight_%"].astype(float)

            # Sort by current weight descending
            df_chart = df_chart.sort_values("Current_weight_%", ascending=False).reset_index(drop=True)

            # Color by type
            type_colors = {
                "ETF": "#1f77b4",
                "Stock": "#ff7f0e",
                "Bond": "#2ca02c",
                "Derivative": "#d62728",
                "Crypto": "#9467bd",
                "Fund": "#8c564b",
                "Other": "#7f7f7f",
            }
            colors = [type_colors.get(t, "#7f7f7f") for t in df_chart["Type"].tolist()]

            # Let Streamlit decide the text color based on theme (light/dark/system)
            text_color = st.get_option("theme.textColor") or "#000000"

            col_current, col_targets = st.columns(2)

            # --- Current allocation chart ---
            with col_current:
                st.markdown("#### Current allocation by asset (% of portfolio)")

                fig_cur, ax_cur = plt.subplots(figsize=(4, max(2, 0.25 * len(df_chart))))
                fig_cur.patch.set_facecolor("none")
                ax_cur.set_facecolor("none")

                y_pos = range(len(df_chart))
                ax_cur.barh(y_pos, df_chart["Current_weight_%"], color=colors)
                ax_cur.set_yticks(y_pos)
                ax_cur.set_yticklabels(df_chart["Asset"], fontsize=7)
                ax_cur.invert_yaxis()  # largest on top
                ax_cur.set_xlabel("Current weight (%)", fontsize=8)

                # Grid and colors
                ax_cur.xaxis.grid(True, linestyle="--", alpha=0.3)
                ax_cur.tick_params(axis="x", colors=text_color, labelsize=7)
                ax_cur.tick_params(axis="y", colors=text_color, labelsize=7)
                ax_cur.xaxis.label.set_color(text_color)
                for spine in ax_cur.spines.values():
                    spine.set_color(text_color)

                # Show value labels at the end of the bars
                for i, (w, val) in enumerate(zip(df_chart["Current_weight_%"], df_chart["Current_value_€"])):
                    ax_cur.text(
                        w + 0.3,
                        i,
                        f"{w:.1f}%\n{val:,.0f}€",
                        va="center",
                        fontsize=6,
                        color=text_color,
                    )

                fig_cur.tight_layout()
                st.pyplot(fig_cur)

            # --- Target vs current comparison chart ---
            with col_targets:
                st.markdown("#### Target vs current weights by asset")

                fig_tar, ax_tar = plt.subplots(figsize=(4, max(2, 0.25 * len(df_chart))))
                fig_tar.patch.set_facecolor("none")
                ax_tar.set_facecolor("none")

                y_pos = range(len(df_chart))
                # Background: target weights (light bar)
                ax_tar.barh(
                    y_pos,
                    df_chart["Target_weight_%"],
                    color=["#dddddd"] * len(df_chart),
                    label="Target",
                )
                # Foreground: current weights (colored by type)
                ax_tar.barh(
                    y_pos,
                    df_chart["Current_weight_%"],
                    color=colors,
                    alpha=0.9,
                    label="Current",
                )

                ax_tar.set_yticks(y_pos)
                ax_tar.set_yticklabels(df_chart["Asset"], fontsize=7)
                ax_tar.invert_yaxis()
                ax_tar.set_xlabel("Weight (%)", fontsize=8)

                ax_tar.xaxis.grid(True, linestyle="--", alpha=0.3)
                ax_tar.tick_params(axis="x", colors=text_color, labelsize=7)
                ax_tar.tick_params(axis="y", colors=text_color, labelsize=7)
                ax_tar.xaxis.label.set_color(text_color)
                for spine in ax_tar.spines.values():
                    spine.set_color(text_color)

                # Simple legend note (text only)
                legend_text = "Grey = Target, Colored = Current (by asset type)"
                ax_tar.text(
                    0.0,
                    -0.8,
                    legend_text,
                    fontsize=7,
                    color=text_color,
                    transform=ax_tar.transAxes,
                )

                fig_tar.tight_layout()
                st.pyplot(fig_tar)
    else:
        st.info("Add assets and set a current value to display the allocation charts.")

    # ============================
    # REBALANCING PLAN (TAB 1)
    # ============================

    st.markdown("---")
    st.subheader("Suggested monthly contribution plan")

    if df_holdings.empty:
        st.info("Add assets to your portfolio and define target weights to compute a contribution plan.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            monthly_contribution = st.number_input(
                "Monthly contribution to invest (€)",
                min_value=0,
                step=50,
                value=500,
                help="Amount you plan to invest next month that will be distributed across your assets.",
            )
        with col_c2:
            rebalance_threshold = st.number_input(
                "Max allowed deviation vs target weights (percentage points)",
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                value=2.0,
                help=(
                    "After applying the contribution, if an asset is still above its target weight + this threshold, "
                    "it will be flagged as overweight for potential sales."
                ),
            )

        if monthly_contribution <= 0:
            st.info("Enter a positive monthly contribution to compute the plan.")
        else:
            holdings_map = {
                row["Asset"]: float(row["Current_value_€"])
                for _, row in df_holdings.iterrows()
            }
            targets_map = {
                row["Asset"]: float(row["Target_weight_%"]) / 100.0
                for _, row in df_holdings.iterrows()
            }
            types_map = {
                row["Asset"]: row["Type"]
                for _, row in df_holdings.iterrows()
            }

            port = Portfolio(holdings=holdings_map, targets=targets_map, asset_types=types_map)

            contrib_plan = compute_contribution_plan(
                portfolio=port,
                monthly_contribution=float(monthly_contribution),
                rebalance_threshold=float(rebalance_threshold),
            )

            contrib_plan = {a: float(v) for a, v in contrib_plan.items()}
            contrib_plan = {a: round(v) for a, v in contrib_plan.items()}  # integer euros

            total_before = port.total_value()
            total_after = total_before + sum(contrib_plan.values())

            current_weights = port.current_weights()
            weights_after = {}
            for asset, h0 in holdings_map.items():
                h1 = h0 + contrib_plan.get(asset, 0.0)
                weights_after[asset] = (h1 / total_after) if total_after > 0 else 0.0

            rows_plan = []
            for asset in holdings_map.keys():
                h0 = float(holdings_map[asset])
                w0 = float(current_weights.get(asset, 0.0)) * 100.0
                target_pct = float(targets_map.get(asset, 0.0)) * 100.0
                buy = float(contrib_plan.get(asset, 0.0))
                h1 = h0 + buy
                w1 = float(weights_after.get(asset, 0.0)) * 100.0
                diff_after = w1 - target_pct

                rows_plan.append(
                    {
                        "Asset": asset,
                        "Type": types_map.get(asset, ""),
                        "Current_value_€": round(h0, 2),
                        "Current_weight_%": round(w0, 2),
                        "Target_weight_%": round(target_pct, 2),
                        "Suggested_buy_€": round(buy, 0),
                        "New_value_€": round(h1, 2),
                        "New_weight_%": round(w1, 2),
                        "Deviation_after_pp": round(diff_after, 2),
                    }
                )

            df_plan = pd.DataFrame(rows_plan)

            st.markdown("#### Allocation before and after monthly contribution")
            st.dataframe(df_plan, use_container_width=True)

            st.markdown(
                f"**Total current portfolio value:** {total_before:,.2f} €  "+
                f"**Total after contribution:** {total_after:,.2f} €"
            )

            # ----------------------
            # OPTIONAL SALES SECTION
            # ----------------------

            st.markdown("---")
            st.subheader("Optional sales to reduce overweight positions")

            overweight_assets = [
                r
                for r in rows_plan
                if r["New_weight_%"] > r["Target_weight_%"] + float(rebalance_threshold)
            ]

            if not overweight_assets:
                st.info(
                    "After applying the suggested contribution, no asset exceeds its target weight "
                    "+ threshold. No sales are strictly required from a rebalancing perspective."
                )
            else:
                st.markdown(
                    "The following assets would still be **overweight** after applying the monthly contribution. "
                    "Below is a minimal sales plan that reduces them towards their target + threshold (as upper bound)."
                )

                total_value_sales = total_after
                values_after = {
                    r["Asset"]: float(r["New_value_€"]) for r in rows_plan
                }

                sales = {asset: 0.0 for asset in values_after.keys()}

                overweight_sorted = sorted(
                    overweight_assets,
                    key=lambda r: r["New_weight_%"] - r["Target_weight_%"],
                    reverse=True,
                )

                for r in overweight_sorted:
                    asset = r["Asset"]
                    target_pct = r["Target_weight_%"]
                    max_pct = target_pct + float(rebalance_threshold)
                    v = values_after[asset]

                    if total_value_sales <= 0:
                        break

                    w_current = v / total_value_sales * 100.0
                    if w_current <= max_pct:
                        continue

                    numerator = max_pct * total_value_sales - v * 100.0
                    denominator = max_pct - 100.0
                    if abs(denominator) < 1e-9:
                        continue

                    sale_amount = numerator / denominator
                    if sale_amount <= 0:
                        continue

                    sale_amount = min(sale_amount, v)

                    sales[asset] += sale_amount
                    values_after[asset] -= sale_amount
                    total_value_sales -= sale_amount

                sales = {a: round(v) for a, v in sales.items() if v > 1.0}

                if not sales:
                    st.info(
                        "Overweight positions are mild enough that a sales plan would be very small or unnecessary."
                    )
                else:
                    rows_sales = []
                    total_sales_amount = 0.0
                    for asset, s_amount in sales.items():
                        v_before = next(r["New_value_€"] for r in rows_plan if r["Asset"] == asset)
                        v_after = max(0.0, v_before - s_amount)
                        w_before = v_before / total_after * 100.0 if total_after > 0 else 0.0
                        w_after = v_after / total_value_sales * 100.0 if total_value_sales > 0 else 0.0

                        rows_sales.append(
                            {
                                "Asset": asset,
                                "Type": types_map.get(asset, ""),
                                "Sale_€": round(s_amount, 0),
                                "Value_before_sale_€": round(v_before, 2),
                                "Value_after_sale_€": round(v_after, 2),
                                "Weight_before_sale_%": round(w_before, 2),
                                "Weight_after_sale_%": round(w_after, 2),
                            }
                        )
                        total_sales_amount += s_amount

                    df_sales = pd.DataFrame(rows_sales)

                    st.markdown("#### Suggested sales by asset")
                    st.dataframe(df_sales, use_container_width=True)

                    st.markdown(
                        f"**Total suggested sales:** {total_sales_amount:,.2f} €  "+
                        "(distributed across overweight assets to move closer to targets)."
                    )

with tab2:
    st.header("Compute required monthly contribution for a future goal")

    # If there is a pending long-term plan to load, apply its values
    # to session_state BEFORE instantiating the widgets, so the UI
    # automatically shows the stored configuration.
    pending_plan_lp = st.session_state.pop("pending_plan_lp", None)
    if pending_plan_lp:
        st.session_state["Valor actual de tu cartera invertida (€)"] = pending_plan_lp["current_total"]
        st.session_state["Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)"] = pending_plan_lp["extra_savings"]
        st.session_state["Objetivo de patrimonio futuro que quieres conseguir (€)"] = pending_plan_lp["objetivo_final"]
        st.session_state["Años hasta el objetivo"] = pending_plan_lp["years"]
        st.session_state["Rentabilidad anual estimada (%)"] = pending_plan_lp["annual_return_input"]
        st.session_state["Tener en cuenta impuestos sobre plusvalías al vender todo al final"] = pending_plan_lp["apply_tax"]
        st.session_state["Modo de aportación"] = pending_plan_lp["modo"]
        st.session_state["¿Con cuánto te gustaría empezar aportando cada mes? (€)"] = pending_plan_lp["initial_monthly"]
        st.session_state["¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)"] = pending_plan_lp["salary_pct_input"]

    st.markdown(
        """
Use this section to design a **long-term investment plan** for a specific wealth target.

You can:
- Choose a **future target net worth** (e.g. 50,000 €),
- Decide **in how many years** you want to reach it,
- Assume an **expected annual return** (e.g. 6–8%),
- Let the app compute how much you should contribute:

- Either as a **constant monthly contribution**, or  
- As a **monthly contribution that grows linearly over time**.

You can also include **additional savings you already have** outside this portfolio.
        """
    )

    colA, colB = st.columns(2)

    with colA:
        current_total = st.number_input(
            "Current invested portfolio value (€)",
            min_value=0.0,
            step=100.0,
            value=0.0,
            key="Valor actual de tu cartera invertida (€)",
        )
        extra_savings = st.number_input(
            "Other initial savings to include (cash, emergency fund, etc.) (€)",
            min_value=0.0,
            step=100.0,
            value=0.0,
            key="Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)",
        )
        objetivo_final = st.number_input(
            "Future wealth target you want to reach (€)",
            min_value=0.0,
            step=1000.0,
            value=50000.0,
            key="Objetivo de patrimonio futuro que quieres conseguir (€)",
        )
        years = st.number_input(
            "Years until the target date",
            min_value=1,
            max_value=60,
            step=1,
            value=10,
            key="Años hasta el objetivo",
        )

    with colB:
        annual_return_input = st.number_input(
            "Expected annual return (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=7.0,
            key="Rentabilidad anual estimada (%)",
        )
        annual_return = annual_return_input / 100.0

        apply_tax = st.checkbox(
            "Account for taxes on capital gains when selling everything at the end",
            value=False,
            help=(
                "If enabled, the monthly contribution will be computed so that the final target is NET, "
                "after applying a progressive tax on capital gains at the end of the horizon."
            ),
            key="Tener en cuenta impuestos sobre plusvalías al vender todo al final",
        )

        # We keep the underlying option values in Spanish for backward compatibility
        # with previously saved plans, but display them in English.
        modo = st.radio(
            "Contribution mode",
            options=["Constante", "Creciente"],
            index=0,
            format_func=lambda x: "Constant monthly amount" if x == "Constante" else "Growing monthly amount",
            help=(
                "Constant = the same amount every month. "
                "Growing = you start with one amount and it increases linearly over the years."
            ),
            key="Modo de aportación",
        )

        initial_monthly = 0
        if modo == "Creciente":
            initial_monthly = st.number_input(
                "How much would you like to contribute initially per month? (€)",
                min_value=0,
                step=10,
                value=150,
                key="¿Con cuánto te gustaría empezar aportando cada mes? (€)",
            )

        salary_pct_input = st.number_input(
            "What percentage of your NET salary should the monthly contribution represent? (%) (optional)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=0.0,
            help=(
                "For example, if you want your monthly investment to be 20% of your net salary, "
                "enter 20. This will allow the app to estimate a reference salary."
            ),
            key="¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)",
        )

    if st.button("🧮 Compute plan to reach the goal"):
        if objetivo_final <= 0:
            st.error("The target wealth must be greater than 0.")
        else:
            if modo == "Constante":
                # ------------------------
                # CONSTANT MONTHLY PLAN
                # ------------------------
                months_total = years * 12

                if apply_tax:
                    # We search for the monthly contribution that yields the desired NET amount
                    def net_final_with_monthly(C: float):
                        C_int = int(round(C))
                        if C_int < 0:
                            C_int = 0
                        final_value_sim, _ = simulate_constant_plan(
                            current_total=current_total,
                            monthly_contribution=C_int,
                            years=years,
                            annual_return=annual_return,
                            extra_savings=extra_savings,
                        )
                        principal_total_sim = current_total + extra_savings + C_int * months_total
                        gain_sim = max(0.0, final_value_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_value_sim - tax_sim
                        return net_final_sim, final_value_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_with_monthly(0.0)
                    if net0 >= objetivo_final:
                        mensual_necesaria = 0
                        final_value = current_total + extra_savings
                        gain = 0.0
                        tax = 0.0
                        net_final = final_value
                        series = [final_value] * months_total
                    else:
                        low = 0.0
                        high = max(objetivo_final / max(months_total, 1) * 3, 5000.0)
                        final_value = 0.0
                        gain = 0.0
                        tax = 0.0
                        net_final = 0.0
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_with_monthly(mid)
                            if net_mid < objetivo_final:
                                low = mid
                            else:
                                high = mid
                                final_value = final_mid
                                gain = gain_mid
                                tax = tax_mid
                                net_final = net_mid
                        mensual_necesaria = int(round(high))

                    # Re-simulate to obtain the full (gross) time series
                    final_value, series = simulate_constant_plan(
                        current_total=current_total,
                        monthly_contribution=mensual_necesaria,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                    )
                    principal_total = current_total + extra_savings + mensual_necesaria * months_total
                    gain = max(0.0, final_value - principal_total)
                    tax = compute_progressive_tax(gain)
                    net_final = final_value - tax

                else:
                    # No tax: use the original helper
                    mensual_necesaria = required_constant_monthly_for_goal(
                        current_total=current_total,
                        objective_final=objetivo_final,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                        tax_rate=0.0,
                    )
                    final_value, series = simulate_constant_plan(
                        current_total=current_total,
                        monthly_contribution=mensual_necesaria,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                    )
                    months_total = years * 12
                    principal_total = current_total + extra_savings + mensual_necesaria * months_total
                    gain = max(0.0, final_value - principal_total)
                    tax = 0.0
                    net_final = final_value

                if mensual_necesaria == 0:
                    st.success(
                        "With your current capital and the assumed return, "
                        "you would theoretically reach the target without needing extra monthly contributions."
                    )
                else:
                    st.subheader("📌 Result (constant contribution)")
                    st.write(
                        f"To reach **{objetivo_final:,.0f} € NET** in **{years} years** "
                        f"with an expected annual return of **{annual_return_input:.1f}%**, "
                        f"you should invest approximately **{mensual_necesaria} € per month**, "
                        "kept constant over the whole period."
                    )

                st.write(
                    f"Estimated gross portfolio value at the end: **{final_value:,.0f} €**"
                )
                st.write(
                    f"Capital gain (before taxes): **{gain:,.0f} €**"
                )
                if apply_tax:
                    st.write(
                        "Estimated taxes on capital gains (using progressive brackets): "
                        f"**{tax:,.0f} €**"
                    )
                    st.write(
                        f"Estimated NET wealth after taxes: **{net_final:,.0f} €**"
                    )

                # Salary reference calculation
                if salary_pct_input > 0 and mensual_necesaria > 0:
                    pct = salary_pct_input / 100.0
                    sueldo_bruto_anual = mensual_necesaria * 12 / pct
                    sueldo_neto_anual, ss_contrib, irpf, eff_rate = compute_salary_net(sueldo_bruto_anual)

                    st.markdown("#### 💼 Reference salary for that monthly contribution")
                    st.write(
                        f"For **{mensual_necesaria} € per month** to represent approximately **{salary_pct_input:.0f}%** "
                        f"of your NET salary, you would need a reference gross salary of about "
                        f"**{sueldo_bruto_anual:,.0f} € per year**, which translates into "
                        f"~**{sueldo_neto_anual:,.0f} € net per year** after an estimated total tax + social security "
                        f"burden of **{eff_rate*100:.1f}%**."
                    )

                    st.caption(
                        "This net salary estimation is approximate. It uses generic IRPF brackets and a ~6.35% "
                        "employee Social Security contribution, without considering specific allowances or deductions."
                    )

                st.markdown("#### Estimated portfolio evolution (gross, before taxes)")
                df_evol = pd.DataFrame(
                    {
                        "Year": [m / 12 for m in range(1, len(series) + 1)],
                        "Estimated_portfolio_€": series,
                    }
                )
                st.line_chart(df_evol, x="Year", y="Estimated_portfolio_€")

                st.caption(
                    "This is a simple simulation of the **gross portfolio value month by month**. "
                    "It does not take into account market volatility, changing tax regimes, or variable returns."
                )

            else:  # modo == "Creciente"
                # ------------------------
                # GROWING MONTHLY PLAN
                # ------------------------
                if initial_monthly <= 0:
                    st.error("The initial monthly contribution must be greater than 0 for a growing plan.")
                else:
                    months_total = years * 12

                    if apply_tax:
                        # We search for the final monthly contribution so that the NET final value hits the target
                        def net_final_with_final_monthly(F: float):
                            F_float = float(F)
                            final_val, _ = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=F_float,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + F_float) / 2.0
                            principal_total_sim = current_total + extra_savings + contrib_total
                            gain_sim = max(0.0, final_val - principal_total_sim)
                            tax_sim = compute_progressive_tax(gain_sim)
                            net_final_sim = final_val - tax_sim
                            return net_final_sim, final_val, gain_sim, tax_sim

                        net0, _, _, _ = net_final_with_final_monthly(initial_monthly)
                        if net0 >= objetivo_final:
                            final_monthly_aprox = initial_monthly
                            final_value_grow, series_grow = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=final_monthly_aprox,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                            principal_total = current_total + extra_savings + contrib_total
                            gain = max(0.0, final_value_grow - principal_total)
                            tax = compute_progressive_tax(gain)
                            net_final = final_value_grow - tax
                        else:
                            low = initial_monthly
                            high = max(initial_monthly * 3, 5000.0)
                            final_value_grow = 0.0
                            gain = 0.0
                            tax = 0.0
                            net_final = 0.0
                            series_grow = []
                            for _ in range(40):
                                mid = (low + high) / 2
                                net_mid, final_mid, gain_mid, tax_mid = net_final_with_final_monthly(mid)
                                if net_mid < objetivo_final:
                                    low = mid
                                else:
                                    high = mid
                                    final_value_grow = final_mid
                                    gain = gain_mid
                                    tax = tax_mid
                                    net_final = net_mid
                            final_monthly_aprox = int(round(high))
                            final_value_grow, series_grow = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=final_monthly_aprox,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                            principal_total = current_total + extra_savings + contrib_total
                            gain = max(0.0, final_value_grow - principal_total)
                            tax = compute_progressive_tax(gain)
                            net_final = final_value_grow - tax
                    else:
                        final_monthly_aprox, resumen_anual = required_growing_monthlies_for_goal(
                            current_total=current_total,
                            objective_final=objetivo_final,
                            years=years,
                            annual_return=annual_return,
                            initial_monthly=initial_monthly,
                            extra_savings=extra_savings,
                            tax_rate=0.0,
                        )
                        final_value_grow, series_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly,
                            final_monthly=final_monthly_aprox,
                            years=years,
                            annual_return=annual_return,
                            initial_value=current_total + extra_savings,
                        )
                        contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                        principal_total = current_total + extra_savings + contrib_total
                        gain = max(0.0, final_value_grow - principal_total)
                        tax = 0.0
                        net_final = final_value_grow

                    # Build an annual summary table
                    resumen_anual = []
                    for año in range(1, years + 1):
                        start_idx = (año - 1) * 12
                        end_idx = año * 12 - 1
                        if months_total > 1:
                            start_month = int(
                                round(
                                    initial_monthly
                                    + (final_monthly_aprox - initial_monthly) * (start_idx / (months_total - 1))
                                )
                            )
                            end_month = int(
                                round(
                                    initial_monthly
                                    + (final_monthly_aprox - initial_monthly) * (end_idx / (months_total - 1))
                                )
                            )
                        else:
                            start_month = final_monthly_aprox
                            end_month = final_monthly_aprox
                        avg_month = int(round((start_month + end_month) / 2))
                        resumen_anual.append(
                            {
                                "year": año,
                                "start": start_month,
                                "end": end_month,
                                "avg": avg_month,
                            }
                        )

                    st.subheader("📌 Result (growing contributions)")
                    st.write(
                        f"To reach approximately **{objetivo_final:,.0f} € NET** in **{years} years** "
                        f"with an expected annual return of **{annual_return_input:.1f}%** and growing contributions, "
                        f"you should start contributing **{initial_monthly} € per month** and end up contributing "
                        f"around **{final_monthly_aprox} € per month**."
                    )

                    df_resumen = pd.DataFrame(resumen_anual)
                    df_resumen = df_resumen.rename(
                        columns={
                            "year": "Year",
                            "start": "Start_€/month",
                            "end": "End_€/month",
                            "avg": "Average_€/month",
                        }
                    )

                    # If the user specified a salary percentage, add derived gross and net salary columns
                    if salary_pct_input > 0:
                        pct = salary_pct_input / 100.0
                        sueldos_brutos = []
                        sueldos_netos = []
                        retenciones_totales = []
                        for _, fila in df_resumen.iterrows():
                            media_mes = fila["Average_€/month"]
                            if media_mes <= 0:
                                sueldo_bruto_anual = 0.0
                                sueldo_neto_anual = 0.0
                                ret_total_pct = 0.0
                            else:
                                sueldo_bruto_anual = media_mes * 12 / pct
                                sueldo_neto_anual, ss_contrib, irpf, eff_rate = compute_salary_net(sueldo_bruto_anual)
                                ret_total_pct = eff_rate * 100.0
                            sueldos_brutos.append(round(sueldo_bruto_anual))
                            sueldos_netos.append(round(sueldo_neto_anual))
                            retenciones_totales.append(round(ret_total_pct, 1))

                        df_resumen["Required_gross_salary_€/year"] = sueldos_brutos
                        df_resumen["Estimated_net_salary_€/year"] = sueldos_netos
                        df_resumen["Approx_total_withholding_%"] = retenciones_totales

                    st.markdown("#### Approximate contributions per year")
                    st.dataframe(df_resumen)

                    st.markdown(
                        "Each row represents one year of the plan:\n"
                        "- **Start_€/month**: monthly contribution at the beginning of that year.\n"
                        "- **End_€/month**: monthly contribution at the end of that year.\n"
                        "- **Average_€/month**: approximate average monthly contribution during that year.\n"
                        "- **Required_gross_salary_€/year** (if a salary % was provided): gross salary so that the average monthly contribution represents that percentage."
                    )

                    st.write(
                        f"Estimated gross portfolio value at the end: **{final_value_grow:,.0f} €**"
                    )
                    st.write(
                        f"Capital gain (before taxes): **{gain:,.0f} €**"
                    )
                    if apply_tax:
                        st.write(
                            "Estimated taxes on capital gains (using progressive brackets): "
                            f"**{tax:,.0f} €**"
                        )
                        st.write(
                            f"Estimated NET wealth after taxes: **{net_final:,.0f} €**"
                        )

                    st.markdown("#### Estimated portfolio evolution (gross, before taxes)")
                    df_evol_grow = pd.DataFrame(
                        {
                            "Year": [m / 12 for m in range(1, len(series_grow) + 1)],
                            "Estimated_portfolio_€": series_grow,
                        }
                    )
                    st.line_chart(df_evol_grow, x="Year", y="Estimated_portfolio_€")

                    st.caption(
                        "This is a simple simulation of the **gross portfolio value month by month** with growing contributions. "
                        "It does not model real market volatility, changing tax regimes, or variable returns."
                    )

                    st.caption(
                        "The monthly contribution increases linearly between the initial and final amounts. "
                        "It does not capture real-world salary jumps or changes, but it helps visualize the trend."
                    )

    # --------------------------
    # Saved long-term plans
    # --------------------------
    st.markdown("---")
    st.markdown("### 💾 Saved long-term plans")

    plans = load_plans()
    planes_lp = plans.get("largo_plazo", {})

    col_plan_lp_1, col_plan_lp_2 = st.columns([2, 2])
    with col_plan_lp_1:
        nombre_plan_lp = st.text_input(
            "Name for this long-term plan",
            value="",
        )
    with col_plan_lp_2:
        opciones_planes_lp = ["(none)"] + sorted(planes_lp.keys()) if isinstance(planes_lp, dict) else ["(none)"]
        plan_lp_seleccionado = st.selectbox(
            "Load an existing plan",
            options=opciones_planes_lp,
        )

    col_plan_lp_save, col_plan_lp_load = st.columns(2)
    with col_plan_lp_save:
        if st.button("💾 Save long-term plan"):
            if not nombre_plan_lp:
                st.error("Please enter a name for the plan before saving.")
            else:
                if not isinstance(plans.get("largo_plazo"), dict):
                    plans["largo_plazo"] = {}
                plans["largo_plazo"][nombre_plan_lp] = {
                    "current_total": current_total,
                    "extra_savings": extra_savings,
                    "objetivo_final": objetivo_final,
                    "years": int(years),
                    "annual_return_input": annual_return_input,
                    "apply_tax": apply_tax,
                    "modo": modo,
                    "initial_monthly": initial_monthly,
                    "salary_pct_input": salary_pct_input,
                }
                save_plans(plans)
                st.success(f"Plan '{nombre_plan_lp}' saved successfully.")
    with col_plan_lp_load:
        if st.button("📂 Load long-term plan"):
            if plan_lp_seleccionado == "(none)":
                st.warning("Select a plan to load.")
            else:
                plan = planes_lp.get(plan_lp_seleccionado)
                if not plan:
                    st.error("The selected plan could not be loaded.")
                else:
                    # Store the selected plan as 'pending' and rerun;
                    # on the next run, it will be applied to session_state before widgets are created.
                    st.session_state["pending_plan_lp"] = plan
                    st.rerun()

    # --- Reset TAB 2 ---
    st.markdown("---")
    if st.button("🔄 Reset this tab", key="reset_tab2"):
        keys_lp = [
            "Valor actual de tu cartera invertida (€)",
            "Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)",
            "Objetivo de patrimonio futuro que quieres conseguir (€)",
            "Años hasta el objetivo",
            "Rentabilidad anual estimada (%)",
            "Tener en cuenta impuestos sobre plusvalías al vender todo al final",
            "Modo de aportación",
            "¿Con cuánto te gustaría empezar aportando cada mes? (€)",
            "¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)",
        ]
        for key in keys_lp:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================
# TAB 3: PLAN DE VIVIENDA
# ============================
with tab3:
    st.header("Plan de vivienda (ahorro para la entrada)")

    # Si hay un plan de vivienda pendiente de cargar, volcamos sus valores ANTES de instanciar los widgets
    pending_plan_viv = st.session_state.pop("pending_plan_viv", None)
    if pending_plan_viv:
        st.session_state["Precio estimado de la vivienda (€)"] = pending_plan_viv["house_price"]
        st.session_state["% de entrada que exige el banco (%)"] = pending_plan_viv["entrada_pct"]
        st.session_state["Años hasta la compra"] = pending_plan_viv["years_house"]
        st.session_state["Ahorro ya destinado a la entrada (€)"] = pending_plan_viv["ahorro_actual_entrada"]
        st.session_state["Rentabilidad anual estimada del ahorro para la entrada (%)"] = pending_plan_viv["anual_return_house_input"]

    st.markdown(
        """
Este modo está pensado para planificar **la entrada de una vivienda**.

1. Indicas el precio objetivo de la vivienda y el % de entrada (por ejemplo, 20%).  
2. Indicas los **gastos asociados** (ITP, notaría, gestoría, etc.) como porcentaje sobre el precio.  
3. Dices cuántos años faltan hasta la compra y cuánto tienes ya ahorrado para la entrada.  
4. Asumes una rentabilidad anual para el dinero que destines a este objetivo.  
5. Eliges **aportación constante** o **aportación creciente**.  
6. Opcionalmente, puedes activar el cálculo de **impuestos sobre plusvalías** al vender la hucha al final.  
7. Además, puedes simular una **hipoteca** (tipo y plazo) para ver la cuota aproximada.

La app te indica cuánto ahorrar, cuánto aportar al mes, en cuánto tiempo y te da una recomendación de cartera para este objetivo.
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        house_price = st.number_input(
            "Precio estimado de la vivienda (€)",
            min_value=0.0,
            step=5000.0,
            value=200000.0,
            key="Precio estimado de la vivienda (€)",
        )
        entrada_pct = st.number_input(
            "% de entrada que exige el banco (%)",
            min_value=0.0,
            max_value=60.0,
            step=1.0,
            value=20.0,
            key="% de entrada que exige el banco (%)",
        )
        gastos_pct = st.number_input(
            "Gastos asociados (ITP, notaría, gestoría, etc.) sobre el precio (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=10.0,
            key="Gastos asociados vivienda (%)",
        )
        years_house = st.number_input(
            "Años hasta la compra",
            min_value=1,
            max_value=40,
            step=1,
            value=7,
            key="Años hasta la compra",
        )

    with col2:
        ahorro_actual_entrada = st.number_input(
            "Ahorro ya destinado a la entrada (€)",
            min_value=0.0,
            step=1000.0,
            value=0.0,
            key="Ahorro ya destinado a la entrada (€)",
        )
        anual_return_house_input = st.number_input(
            "Rentabilidad anual estimada del ahorro para la entrada (%)",
            min_value=0.0,
            max_value=15.0,
            step=0.5,
            value=4.0,
            key="Rentabilidad anual estimada del ahorro para la entrada (%)",
        )
        annual_return_house = anual_return_house_input / 100.0

        apply_tax_house = st.checkbox(
            "Tener en cuenta impuestos sobre plusvalías al vender la hucha al final",
            value=False,
            help=(
                "Si lo marcas, la cuota mensual se calculará para que el efectivo objetivo sea NETO, "
                "después de pagar impuestos sobre las plusvalías con tramos progresivos."
            ),
            key="Tener en cuenta impuestos vivienda",
        )

        tipo_hipoteca_input = st.number_input(
            "Tipo de interés anual aproximado de la hipoteca (%)",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=3.0,
            key="Tipo interés hipoteca",
        )
        plazo_hipoteca_years = st.number_input(
            "Plazo de la hipoteca (años)",
            min_value=1,
            max_value=40,
            step=1,
            value=30,
            key="Plazo hipoteca años",
        )

    # --- Elegir modo de aportación (constante/creciente) para la entrada
    modo_house = st.radio(
        "Modo de aportación para la entrada",
        options=["Constante", "Creciente"],
        index=0,
        help="Constante = mismo importe todos los meses. Creciente = empiezas con una cantidad y vas subiendo cada año.",
        key="Modo de aportación vivienda",
    )

    initial_monthly_house = 0
    if modo_house == "Creciente":
        initial_monthly_house = st.number_input(
            "¿Con cuánto te gustaría empezar aportando cada mes para la entrada? (€)",
            min_value=0,
            step=50,
            value=300,
            key="Aportación inicial vivienda",
        )

    entrada_objetivo = house_price * entrada_pct / 100.0
    gastos_totales = house_price * gastos_pct / 100.0
    objetivo_total_efectivo = entrada_objetivo + gastos_totales
    restante_necesario = max(0.0, objetivo_total_efectivo - ahorro_actual_entrada)

    st.markdown(
        f"Entrada objetivo: **{entrada_objetivo:,.0f} €** (~{entrada_pct:.0f}% de {house_price:,.0f} €)."
    )
    st.markdown(
        f"Gastos asociados estimados: **{gastos_totales:,.0f} €** (~{gastos_pct:.1f}% sobre el precio)."
    )
    st.markdown(
        f"Total de efectivo objetivo (entrada + gastos): **{objetivo_total_efectivo:,.0f} €**."
    )
    st.markdown(
        f"De ese total, te faltan por ahorrar aproximadamente **{restante_necesario:,.0f} €**."
    )

    # Simulación rápida de hipoteca
    hipoteca_principal = max(0.0, house_price - entrada_objetivo)
    if hipoteca_principal > 0 and tipo_hipoteca_input >= 0 and plazo_hipoteca_years > 0:
        r_mensual = (tipo_hipoteca_input / 100.0) / 12.0
        n_meses_hipoteca = int(plazo_hipoteca_years * 12)
        if r_mensual > 0:
            cuota_mensual_hipoteca = hipoteca_principal * r_mensual * (1 + r_mensual) ** n_meses_hipoteca / (
                (1 + r_mensual) ** n_meses_hipoteca - 1
            )
        else:
            cuota_mensual_hipoteca = hipoteca_principal / n_meses_hipoteca
        st.markdown(
            f"💳 Hipoteca simulada: principal aproximado **{hipoteca_principal:,.0f} €**, "
            f"cuota mensual estimada **{cuota_mensual_hipoteca:,.0f} €** a {plazo_hipoteca_years:.0f} años "
            f"con un tipo del {tipo_hipoteca_input:.1f}%."
        )

    if st.button("🏠 Calcular plan de ahorro para la entrada"):
        if house_price <= 0 or entrada_pct <= 0:
            st.error("Introduce un precio de vivienda y un porcentaje de entrada mayores que 0.")
        elif years_house <= 0:
            st.error("Los años hasta la compra deben ser mayores que 0.")
        elif restante_necesario <= 0:
            st.success(
                "Con lo que ya tienes ahorrado para la entrada, en principio llegarías al objetivo sin necesidad de aportar más."
            )
        elif modo_house == "Creciente" and initial_monthly_house <= 0:
            st.error("La aportación inicial para la entrada debe ser mayor que 0 si eliges modo creciente.")
        else:
            months_house = int(years_house) * 12

            if modo_house == "Constante":
                # === MODO CONSTANTE ===
                if apply_tax_house:
                    # Buscamos la aportación mensual para que el objetivo total (entrada + gastos) sea NETO tras impuestos
                    def net_final_house_with_monthly(C: float):
                        C_int = int(round(C))
                        if C_int < 0:
                            C_int = 0
                        final_val_sim, _ = simulate_constant_plan(
                            current_total=ahorro_actual_entrada,
                            monthly_contribution=C_int,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            extra_savings=0.0,
                        )
                        principal_total_sim = ahorro_actual_entrada + C_int * months_house
                        gain_sim = max(0.0, final_val_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_val_sim - tax_sim
                        return net_final_sim, final_val_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_house_with_monthly(0.0)
                    if net0 >= objetivo_total_efectivo:
                        mensual_entrada = 0
                        final_entrada = ahorro_actual_entrada
                        principal_total_entrada = ahorro_actual_entrada
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = final_entrada
                        series_entrada = [final_entrada] * months_house
                    else:
                        low = 0.0
                        high = max(objetivo_total_efectivo / max(months_house, 1) * 3, 5000.0)
                        final_entrada = 0.0
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = 0.0
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_house_with_monthly(mid)
                            if net_mid < objetivo_total_efectivo:
                                low = mid
                            else:
                                high = mid
                                final_entrada = final_mid
                                gain_entrada = gain_mid
                                tax_entrada = tax_mid
                                net_final_entrada = net_mid
                        mensual_entrada = int(round(high))
                        final_entrada, series_entrada = simulate_constant_plan(
                            current_total=ahorro_actual_entrada,
                            monthly_contribution=mensual_entrada,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            extra_savings=0.0,
                        )
                        principal_total_entrada = ahorro_actual_entrada + mensual_entrada * months_house
                        gain_entrada = max(0.0, final_entrada - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada - tax_entrada
                else:
                    # Sin impuestos: objetivo total bruto (entrada + gastos)
                    mensual_entrada = required_constant_monthly_for_goal(
                        current_total=ahorro_actual_entrada,
                        objetivo_final=objetivo_total_efectivo,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        extra_savings=0.0,
                        tax_rate=0.0,
                    )
                    final_entrada, series_entrada = simulate_constant_plan(
                        current_total=ahorro_actual_entrada,
                        monthly_contribution=mensual_entrada,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        extra_savings=0.0,
                    )
                    principal_total_entrada = ahorro_actual_entrada + mensual_entrada * months_house
                    gain_entrada = max(0.0, final_entrada - principal_total_entrada)
                    tax_entrada = 0.0
                    net_final_entrada = final_entrada

                st.subheader("📌 Plan de ahorro para la entrada (aportación constante)")
                objetivo_texto = "NETOS" if apply_tax_house else "brutos"
                st.write(
                    f"Para llegar a un efectivo total de **{objetivo_total_efectivo:,.0f} € {objetivo_texto}** "
                    f"(entrada + gastos) en **{int(years_house)} años**, "
                    f"con una rentabilidad anual estimada del **{anual_return_house_input:.1f}%**, "
                    f"deberías ahorrar/invertir aproximadamente **{mensual_entrada} € al mes** dedicados a este objetivo."
                )

                st.write(
                    f"Patrimonio bruto estimado en la hucha al final: **{final_entrada:,.0f} €**"
                )
                st.write(
                    f"Aportaciones totales realizadas: **{principal_total_entrada:,.0f} €**"
                )
                st.write(
                    f"Plusvalía estimada (beneficio antes de impuestos): **{gain_entrada:,.0f} €**"
                )
                if apply_tax_house:
                    st.write(
                        f"Impuestos estimados sobre plusvalías (según tramos progresivos): **{tax_entrada:,.0f} €**"
                    )
                    st.write(
                        f"Efectivo neto estimado tras impuestos: **{net_final_entrada:,.0f} €**"
                    )

                st.markdown("#### Evolución estimada del ahorro para la entrada")
                df_entrada = pd.DataFrame(
                    {
                        "Año": [m / 12 for m in range(1, len(series_entrada) + 1)],
                        "Ahorro_estimado_€": series_entrada,
                    }
                )
                st.line_chart(df_entrada, x="Año", y="Ahorro_estimado_€")

                st.caption(
                    "Es una simulación sencilla del crecimiento de la 'hucha' para la entrada, "
                    "suponiendo aportaciones constantes y una rentabilidad media estable."
                )

            else:
                # === MODO CRECIENTE ===
                if apply_tax_house:
                    # Buscamos la aportación final mensual para que el efectivo total sea NETO tras impuestos
                    def net_final_house_with_final_monthly(F: float):
                        F_float = float(F)
                        final_val_sim, _ = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=F_float,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total_sim = months_house * (initial_monthly_house + F_float) / 2.0
                        principal_total_sim = ahorro_actual_entrada + contrib_total_sim
                        gain_sim = max(0.0, final_val_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_val_sim - tax_sim
                        return net_final_sim, final_val_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_house_with_final_monthly(initial_monthly_house)
                    if net0 >= objetivo_total_efectivo:
                        final_monthly_house = initial_monthly_house
                        final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=final_monthly_house,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                        principal_total_entrada = ahorro_actual_entrada + contrib_total
                        gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada_grow - tax_entrada
                    else:
                        low = initial_monthly_house
                        high = max(initial_monthly_house * 3, 5000.0)
                        final_entrada_grow = 0.0
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = 0.0
                        series_entrada_grow = []
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_house_with_final_monthly(mid)
                            if net_mid < objetivo_total_efectivo:
                                low = mid
                            else:
                                high = mid
                                final_entrada_grow = final_mid
                                gain_entrada = gain_mid
                                tax_entrada = tax_mid
                                net_final_entrada = net_mid
                        final_monthly_house = int(round(high))
                        final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=final_monthly_house,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                        principal_total_entrada = ahorro_actual_entrada + contrib_total
                        gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada_grow - tax_entrada
                else:
                    # Sin impuestos: objetivo total bruto
                    final_monthly_house, _ = required_growing_monthlies_for_goal(
                        current_total=ahorro_actual_entrada,
                        objetivo_final=objetivo_total_efectivo,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        initial_monthly=initial_monthly_house,
                        extra_savings=0.0,
                        tax_rate=0.0,
                    )
                    final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                        initial_monthly=initial_monthly_house,
                        final_monthly=final_monthly_house,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        initial_value=ahorro_actual_entrada,
                    )
                    contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                    principal_total_entrada = ahorro_actual_entrada + contrib_total
                    gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                    tax_entrada = 0.0
                    net_final_entrada = final_entrada_grow

                # Construimos resumen anual de aportaciones
                resumen_anual_house = []
                for año in range(1, int(years_house) + 1):
                    start_idx = (año - 1) * 12
                    end_idx = año * 12 - 1
                    if months_house > 1:
                        start_month = int(
                            round(
                                initial_monthly_house
                                + (final_monthly_house - initial_monthly_house) * (start_idx / (months_house - 1))
                            )
                        )
                        end_month = int(
                            round(
                                initial_monthly_house
                                + (final_monthly_house - initial_monthly_house) * (end_idx / (months_house - 1))
                            )
                        )
                    else:
                        start_month = final_monthly_house
                        end_month = final_monthly_house
                    avg_month = int(round((start_month + end_month) / 2))
                    resumen_anual_house.append(
                        {
                            "Año": año,
                            "Inicio_€/mes": start_month,
                            "Fin_€/mes": end_month,
                            "Media_€/mes": avg_month,
                        }
                    )

                st.subheader("📌 Plan de ahorro para la entrada (aportación creciente)")
                objetivo_texto = "NETOS" if apply_tax_house else "brutos"
                st.write(
                    f"Para llegar a un efectivo total de **{objetivo_total_efectivo:,.0f} € {objetivo_texto}** "
                    f"(entrada + gastos) en **{int(years_house)} años**, "
                    f"con una rentabilidad anual estimada del **{anual_return_house_input:.1f}%**, "
                    f"deberías empezar aportando **{initial_monthly_house} € al mes** y terminar aportando "
                    f"aproximadamente **{final_monthly_house} € al mes** a este objetivo."
                )

                df_resumen_house = pd.DataFrame(resumen_anual_house)
                st.markdown("#### Aportaciones aproximadas por año (plan entrada vivienda)")
                st.dataframe(df_resumen_house)

                st.markdown(
                    "Cada fila representa un año del plan: \n"
                    "- **Inicio_€/mes**: cuánto aportarías al comienzo de ese año.\n"
                    "- **Fin_€/mes**: cuánto aportarías al final de ese año.\n"
                    "- **Media_€/mes**: aportación mensual media aproximada durante ese año."
                )

                st.write(
                    f"Patrimonio bruto estimado en la hucha al final: **{final_entrada_grow:,.0f} €**"
                )
                st.write(
                    f"con una rentabilidad anual estimada del {anual_return_house_input:.1f}%."
)


# ============================
# TAB 4: ANÁLISIS DE CARTERA
# ============================
with tab4:
    st.header("Análisis de cartera a partir del universo de activos")

    st.markdown(
        """
En esta pestaña puedes construir una **cartera de análisis** eligiendo activos
del universo completo (CSV) y asignándoles un valor en euros.

1. Busca un activo por **nombre**, **ISIN** o cualquier palabra clave.  
2. Añádelo a tu cartera de análisis con un valor actual (€).  
3. Cuando tengas varios activos añadidos, pulsa en **Calcular estadísticas** para ver:
   - Distribución por **región**
   - Distribución por **tipo de activo**
   - Distribución por **divisa**
   - Distribución por **subtipo de ETF** (Equity Global, EM Equity, Bond, etc.)
   - Top 10 posiciones por peso
   - Tabla resumen de la cartera con todos los metadatos relevantes
"""
    )

    # Cargamos universo completo desde el CSV grande
    universe_df = load_universe_csv()
    if universe_df.empty:
        st.error(
            "No se ha podido cargar el universo de activos desde 'asset_universe.csv'. "
            "Asegúrate de que el fichero existe en la misma carpeta que esta app."
        )
    else:
        # Inicializamos la cartera de análisis en sesión
        if "analysis_portfolio" not in st.session_state:
            st.session_state["analysis_portfolio"] = pd.DataFrame(
                columns=[
                    "ISIN",
                    "Name",
                    "Type",
                    "Region",
                    "Country",
                    "ETF_Provider",
                    "ETF_Subtype",
                    "Currency_Name",
                    "Value_€",
                ]
            )

        st.subheader("🔎 Buscar y añadir activos a la cartera de análisis")

        # Desplegable con buscador interno de Streamlit (sin tabla aparte)
        label_df = universe_df.copy()
        label_df["Label"] = label_df.apply(
            lambda r: f"{r.get('Name','')} ({r.get('ISIN','')}) - {r.get('Type','')} {r.get('Region','')}",
            axis=1,
        )

        options = ["(elige un activo)"] + label_df["Label"].tolist()
        selected_label = st.selectbox(
            "Escribe para buscar por nombre/ISIN y selecciona el activo",
            options=options,
            index=0,
            help="Empieza a escribir y usa el buscador interno del desplegable para filtrar.",
        )

        selected_row = None
        if selected_label != "(elige un activo)":
            selected_row = label_df.loc[label_df["Label"] == selected_label].iloc[0]

        col_add1, col_add2 = st.columns(2)
        with col_add1:
            valor_para_anadir = st.number_input(
                "Valor actual (€) a asignar al activo seleccionado",
                min_value=0.0,
                step=100.0,
                value=0.0,
            )
        with col_add2:
            if st.button("➕ Añadir activo a mi cartera de análisis"):
                if selected_row is None:
                    st.error("Primero selecciona un activo de la lista de resultados.")
                elif valor_para_anadir <= 0:
                    st.error("El valor asignado debe ser mayor que 0 €.")
                else:
                    # Construimos una fila con los metadatos relevantes
                    new_row = {
                        "ISIN": selected_row.get("ISIN", ""),
                        "Name": selected_row.get("Name", ""),
                        "Type": selected_row.get("Type", ""),
                        "Region": selected_row.get("Region", ""),
                        "Country": selected_row.get("Country", ""),
                        "ETF_Provider": selected_row.get("ETF_Provider", ""),
                        "ETF_Subtype": selected_row.get("ETF_Subtype", ""),
                        "Currency_Name": selected_row.get("Currency_Name", ""),
                        "Value_€": float(valor_para_anadir),
                    }

                    portfolio_df = st.session_state["analysis_portfolio"].copy()

                    # Si ya existe ese ISIN en la cartera, sumamos al valor existente
                    if not portfolio_df.empty and new_row["ISIN"] in portfolio_df["ISIN"].values:
                        portfolio_df.loc[
                            portfolio_df["ISIN"] == new_row["ISIN"], "Value_€"
                        ] += new_row["Value_€"]
                    else:
                        portfolio_df = pd.concat(
                            [portfolio_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )

                    st.session_state["analysis_portfolio"] = portfolio_df
                    st.success(
                        f"Activo '{new_row['Name']}' añadido/actualizado en la cartera de análisis "
                        f"con {new_row['Value_€']:.2f} €."
                    )

        st.markdown("---")
        st.subheader("📂 Cartera de análisis actual")

        portfolio_df = st.session_state["analysis_portfolio"].copy()

        if portfolio_df.empty:
            st.info("Todavía no hay activos en la cartera de análisis.")
        else:
            # Permitimos editar solo la columna de valor para retocar manualmente
            editable_df = portfolio_df.copy()
            editable_df = st.data_editor(
                editable_df,
                column_config={
                    "Value_€": st.column_config.NumberColumn(
                        "Valor actual (€)",
                        min_value=0.0,
                        step=100.0,
                    )
                },
                disabled=[
                    "ISIN",
                    "Name",
                    "Type",
                    "Region",
                    "Country",
                    "ETF_Provider",
                    "ETF_Subtype",
                    "Currency_Name",
                ],
                use_container_width=True,
                key="analysis_portfolio_editor",
            )

            # Actualizamos sesión con posibles cambios en valores
            st.session_state["analysis_portfolio"] = editable_df
            portfolio_df = editable_df

            total_value = float(portfolio_df["Value_€"].sum())
            st.markdown(f"**Valor total de la cartera de análisis:** {total_value:,.2f} €")

            if st.button("📊 Calcular estadísticas de la cartera de análisis"):
                if total_value <= 0:
                    st.error("El valor total de la cartera debe ser mayor que 0 €.")
                else:
                    # Calculamos pesos
                    portfolio_df = portfolio_df.copy()
                    portfolio_df["Weight_%"] = portfolio_df["Value_€"] / total_value * 100.0

                    st.markdown("### 1️⃣ Top 10 posiciones por peso")
                    top10 = portfolio_df.sort_values("Weight_%", ascending=False).head(10)
                    st.dataframe(
                        top10[
                            ["Name", "ISIN", "Type", "Region", "Currency_Name", "Value_€", "Weight_%"]
                        ],
                        use_container_width=True,
                    )

                    # ==========================
                    # DETECCIÓN TEMA STREAMLIT
                    # ==========================
                    theme_base = st.get_option("theme.base")
                    text_color = st.get_option("theme.textColor")
                    if theme_base == "dark" or not text_color:
                        text_color = "#FFFFFF"
                    else:
                        text_color = text_color or "#000000"

                    # ==========================
                    # 2️⃣ Distribución por región
                    # ==========================
                    st.markdown("### 2️⃣ Distribución por región")
                    region_expo = (
                        portfolio_df.groupby("Region", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    region_expo["Weight_%"] = region_expo["Value_€"] / total_value * 100.0

                    fig_reg, ax_reg = plt.subplots()
                    fig_reg.patch.set_facecolor("none")
                    ax_reg.set_facecolor("none")

                    wedges, texts, autotexts = ax_reg.pie(
                        region_expo["Weight_%"],
                        labels=region_expo["Region"].fillna("Desconocida"),
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    ax_reg.axis("equal")

                    # Textos del pie adaptados al tema
                    for t in texts + autotexts:
                        t.set_color(text_color)

                    st.pyplot(fig_reg)

                    # ==========================
                    # 3️⃣ Distribución por tipo
                    # ==========================
                    st.markdown("### 3️⃣ Distribución por tipo de activo")
                    type_expo = (
                        portfolio_df.groupby("Type", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    type_expo["Weight_%"] = type_expo["Value_€"] / total_value * 100.0

                    fig_type, ax_type = plt.subplots()
                    fig_type.patch.set_facecolor("none")
                    ax_type.set_facecolor("none")

                    ax_type.bar(
                        type_expo["Type"].fillna("Desconocido"),
                        type_expo["Weight_%"],
                    )
                    ax_type.set_ylabel("% de la cartera")
                    ax_type.set_xlabel("Tipo de activo")
                    plt.xticks(rotation=30, ha="right")

                    # Colores de texto y ejes según tema
                    ax_type.tick_params(colors=text_color)
                    ax_type.yaxis.label.set_color(text_color)
                    ax_type.xaxis.label.set_color(text_color)
                    for spine in ax_type.spines.values():
                        spine.set_color(text_color)

                    st.pyplot(fig_type)

                    # ==========================
                    # 4️⃣ Distribución por divisa
                    # ==========================
                    st.markdown("### 4️⃣ Distribución por divisa")
                    currency_expo = (
                        portfolio_df.groupby("Currency_Name", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    currency_expo["Weight_%"] = currency_expo["Value_€"] / total_value * 100.0

                    fig_cur, ax_cur = plt.subplots()
                    fig_cur.patch.set_facecolor("none")
                    ax_cur.set_facecolor("none")

                    ax_cur.bar(
                        currency_expo["Currency_Name"].fillna("Desconocida"),
                        currency_expo["Weight_%"],
                    )
                    ax_cur.set_ylabel("% de la cartera")
                    ax_cur.set_xlabel("Divisa")
                    plt.xticks(rotation=30, ha="right")

                    ax_cur.tick_params(colors=text_color)
                    ax_cur.yaxis.label.set_color(text_color)
                    ax_cur.xaxis.label.set_color(text_color)
                    for spine in ax_cur.spines.values():
                        spine.set_color(text_color)

                    st.pyplot(fig_cur)

                    # ==========================
                    # 5️⃣ Distribución subtipo ETF
                    # ==========================
                    st.markdown("### 5️⃣ Distribución por subtipo de ETF (solo ETFs)")
                    etf_only = portfolio_df[portfolio_df["Type"] == "ETF"].copy()
                    if etf_only.empty:
                        st.info("No hay ETFs en esta cartera de análisis, así que no puede mostrarse esta distribución.")
                    else:
                        etf_sub_expo = (
                            etf_only.groupby("ETF_Subtype", dropna=False)["Value_€"]
                            .sum()
                            .reset_index()
                        )
                        etf_sub_expo["Weight_%"] = etf_sub_expo["Value_€"] / total_value * 100.0

                        fig_sub, ax_sub = plt.subplots()
                        fig_sub.patch.set_facecolor("none")
                        ax_sub.set_facecolor("none")

                        ax_sub.bar(
                            etf_sub_expo["ETF_Subtype"].fillna("Sin clasificar"),
                            etf_sub_expo["Weight_%"],
                        )
                        ax_sub.set_ylabel("% de la cartera")
                        ax_sub.set_xlabel("Subtipo de ETF")
                        plt.xticks(rotation=30, ha="right")

                        ax_sub.tick_params(colors=text_color)
                        ax_sub.yaxis.label.set_color(text_color)
                        ax_sub.xaxis.label.set_color(text_color)
                        for spine in ax_sub.spines.values():
                            spine.set_color(text_color)

                        st.pyplot(fig_sub)
                    

    # --- Reset TAB 4 ---
    st.markdown("---")
    if st.button("🔄 Restablecer análisis", key="reset_tab4"):
        if "analysis_portfolio" in st.session_state:
            del st.session_state["analysis_portfolio"]
        st.rerun()

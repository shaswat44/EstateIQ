"""EstateIQ | Gurgaon property price intelligence (Streamlit)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from packaging.version import Version
from property_filters import property_types_for_bhk_and_flat

try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = None

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gurgaon_price_model_validated.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics_validated.json"

# Use the exact dataset selected for model evaluation
DATA_FILE = BASE_DIR / "data" / "data of gurugram real Estate.csv"

DATA_CANDIDATES = [
    BASE_DIR / "data" / "data of gurugram real Estate.csv",
]

FEATURES = [
    "Area", "Status", "Property Type", "Locality",
    "RERA Approval", "BHK_Count", "Flat Type", "Company Name",
]

# Set True ONLY if the model was trained on log1p(Price).
LOG_TARGET = False

ACCENT = "#48E0C1"
ACCENT_2 = "#7C9CFF"

# `use_container_width` is deprecated in newer Streamlit versions.
STRETCH = (
    {"width": "stretch"}
    if Version(st.__version__) >= Version("1.50.0")
    else {"use_container_width": True}
)

st.set_page_config(
    page_title="EstateIQ | Gurgaon",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. STYLING (base colours live in .streamlit/config.toml)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --border: rgba(148, 163, 184, 0.16);
    --accent: #48E0C1;
}
html, body, .stApp { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(ellipse at 15% 0%, #142943 0%, transparent 38%),
      radial-gradient(ellipse at 100% 20%, #102A30 0%, transparent 30%),
      #080D18;
}
[data-testid="stHeader"] { background: rgba(8,13,24,0.7); }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.6px; }

.hero {
    background: linear-gradient(120deg, rgba(25,48,70,0.94), rgba(14,29,43,0.84));
    border: 1px solid rgba(72,224,193,0.23);
    border-radius: 24px; padding: 30px 32px; margin: 8px 0 25px 0;
    box-shadow: 0 16px 55px rgba(0,0,0,0.16);
}
.hero h1 { font-size: clamp(30px, 4vw, 46px); line-height: 1.12; margin: 12px 0; color: #F8FAFC; }
.hero p { color: #AAB9CD; font-size: 15px; max-width: 720px; margin-bottom: 0; }
.eyebrow { color: #48E0C1; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.page-head { margin: 0 0 18px 0; }
.page-head h2 { margin: 6px 0 4px 0; font-size: 30px; color: #F8FAFC; }
.page-head p { color: #91A0B7; margin: 0; font-size: 14px; }

.glass {
    background: linear-gradient(145deg, rgba(25,38,59,0.92), rgba(14,23,38,0.88));
    border: 1px solid var(--border); border-radius: 18px;
    padding: 20px 22px; box-shadow: 0 12px 35px rgba(0,0,0,0.12); min-height: 125px;
}
.kpi-label { color: #91A0B7; font-size: 12px; margin-bottom: 12px; }
.kpi-value { color: #F8FAFC; font-family: 'Space Grotesk', sans-serif;
             font-size: clamp(22px, 2.1vw, 30px); font-weight: 700; }
.kpi-note { color: #48E0C1; font-size: 12px; margin-top: 8px; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 21px;
                 font-weight: 600; margin: 28px 0 14px 0; color: #F1F5F9; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
         font-size: 11px; font-weight: 700; letter-spacing: .5px; }
.badge-ok { background: rgba(72,224,193,.15); color: #48E0C1; }
.badge-warn { background: rgba(250,176,5,.15); color: #FAB005; }

div[data-testid="stForm"], div[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 18px; background: rgba(17,27,44,0.72);
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid rgba(72,224,193,0.45);
    background: linear-gradient(100deg, #48E0C1, #82F0D5);
    color: #07131B; font-weight: 700; min-height: 44px;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    border-color: #82F0D5; color: #07131B; filter: brightness(1.04);
}
div[data-testid="stMetric"] {
    background: rgba(19,29,48,0.8); border: 1px solid var(--border);
    border-radius: 14px; padding: 15px;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset(path_string: str, modified_time: float) -> pd.DataFrame:
    data = pd.read_csv(path_string)
    data.columns = data.columns.str.strip()

    # Convert numeric columns safely
    for col in ("Price", "Area", "BHK_Count", "Rate per sqft"):
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    # Clean categorical columns
    categorical_cols = [
        "Locality", "Flat Type", "Status",
        "Property Type", "Company Name", "RERA Approval",
        "Builder Name", "Socity",
    ]

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str).str.strip()
            data[col] = data[col].replace({"nan": "", "None": ""})

    # Remove rows without a valid target price
    if "Price" in data.columns:
        data = data.dropna(subset=["Price"])
        data = data[data["Price"] > 0].copy()

    # Derived metric for dashboard analytics only
    if {"Price", "Area"} <= set(data.columns):
        data["Price_per_sqft"] = (
            data["Price"] / data["Area"].where(data["Area"] > 0)
        )

    return data


@st.cache_data
def load_metrics(path_string: str, modified_time: float) -> dict:
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(f"Check that {MODEL_PATH.name} exists inside the models folder.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

df = None
data_error = None
EXISTING_DATA = [p for p in DATA_CANDIDATES if p.exists()]
DATA_PATH = Path(DATA_FILE) if DATA_FILE else (EXISTING_DATA[0] if EXISTING_DATA else None)
if DATA_PATH is not None and not DATA_PATH.exists():
    data_error = f"Dataset file not found: {DATA_PATH}"
    DATA_PATH = None
if DATA_PATH is not None:
    try:
        df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        data_error = str(exc)


@st.cache_data
def model_fingerprint(path_string: str, modified_time: float) -> str:
    h = hashlib.sha256()
    with open(path_string, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MODEL_SHA = model_fingerprint(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
DATA_SHA = model_fingerprint(str(DATA_PATH), DATA_PATH.stat().st_mtime) if DATA_PATH else None

metrics = None
metrics_status = "missing"   # missing | verified | unverified | mismatch
metrics_notes: list[str] = []
if METRICS_PATH.exists():
    try:
        metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
    except Exception:
        metrics = None
    if metrics is not None:
        recorded = metrics.get("model_sha256")
        if recorded and recorded != MODEL_SHA:
            metrics, metrics_status = None, "mismatch"
        elif recorded:
            metrics_status = "verified"
        else:
            metrics_status = "unverified"
        trained_with = (metrics or {}).get("sklearn_version")
        if trained_with and SKLEARN_VERSION and trained_with != SKLEARN_VERSION:
            metrics_notes.append(
                f"Model was evaluated with scikit-learn {trained_with} but this environment "
                f"runs {SKLEARN_VERSION}. Pin the training version to avoid subtle differences.")
        rec_data = (metrics or {}).get("data_sha256")
        if metrics is not None and rec_data and DATA_SHA and rec_data != DATA_SHA:
            metrics_notes.append(
                "The dashboard CSV is not the same file these metrics were evaluated on. "
                "Comparisons use this CSV, so re-run evaluate_model.py if the data changed.")


# --------------------------------------------------
# 4. HELPERS
# --------------------------------------------------
def money_inr(value) -> str:
    value = float(value)
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:,.2f} L"
    return f"₹{value:,.0f}"


def money_full(value) -> str:
    return f"₹{float(value):,.0f}"


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">ESTATEIQ</div>'
        f"<h2>{title}</h2><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def glass_card(label, value, note) -> None:
    st.markdown(
        f"""<div class="glass"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int | None = None, title: str | None = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="DM Sans"),
        title_font_size=16,
    )
    if title:
        fig.update_layout(title=title)
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def show(fig, **kw) -> None:
    st.plotly_chart(fig, **STRETCH, **kw)


def options_from_data(column: str, fallback: list[str]) -> list[str]:
    if df is not None and column in df.columns:
        values = df[column].dropna().astype(str).str.strip()
        values = sorted(v for v in values.unique() if v and v.lower() != "nan")
        if values:
            return values
    return fallback


def reset_invalid_property_type():
    bhk = st.session_state.get("bhk_count", 0)
    flat_type = st.session_state.get("flat_type", "")

    valid_options = property_types_for_bhk_and_flat(df, bhk, flat_type)

    current = st.session_state.get("property_type")

    if valid_options:
        if current not in valid_options:
            st.session_state["property_type"] = valid_options[0]
    else:
        st.session_state.pop("property_type", None)


def default_index(options: list[str], preferred) -> int:
    if isinstance(preferred, str):
        preferred = [preferred]
    for p in preferred:
        if p in options:
            return options.index(p)
    return 0


def price_hist(series: pd.Series, nbins: int, log: bool, title: str | None = None):
    series = series[series > 0].dropna()
    if log:
        x = np.log10(series)
        fig = px.histogram(x=x, nbins=nbins, labels={"x": "Listed price (log scale)"})
        ticks = [t for t in (1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9)
                 if x.min() - 0.3 <= np.log10(t) <= x.max() + 0.3]
        fig.update_xaxes(tickvals=[np.log10(t) for t in ticks],
                         ticktext=[money_inr(t) for t in ticks])
    else:
        fig = px.histogram(x=series, nbins=nbins, labels={"x": "Listed price (₹)"})
    fig.update_traces(marker_color=ACCENT)
    return style_fig(fig, title=title)


# ---- model helpers ----
def final_estimator(m):
    return m.steps[-1][1] if hasattr(m, "steps") else m


def preprocessor(m):
    return m[:-1] if hasattr(m, "steps") and len(m.steps) > 1 else None


def point_predict(m, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(m.predict(X), dtype=float)
    return np.expm1(pred) if LOG_TARGET else pred


def predict_range(m, X: pd.DataFrame, low_q: int = 10, high_q: int = 90):
    """Spread of individual tree predictions. Returns ((low, high) | None, reason | None)."""
    est = final_estimator(m)
    if not hasattr(est, "estimators_"):
        return None, "the final model is not an ensemble of trees"
    try:
        pre = preprocessor(m)
        Xt = pre.transform(X) if pre is not None else X
        preds = np.array([t.predict(Xt)[0] for t in est.estimators_])
        if LOG_TARGET:
            preds = np.expm1(preds)
        low, high = np.percentile(preds, [low_q, high_q])
        return (float(low), float(high)), None
    except Exception as exc:
        return None, f"could not read per-tree predictions ({exc})"


def feature_importance(m) -> pd.Series | None:
    est, pre = final_estimator(m), preprocessor(m)
    if pre is None or not hasattr(est, "feature_importances_"):
        return None
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return None
    ordered = sorted(FEATURES, key=len, reverse=True)

    def origin(name: str) -> str:
        name = name.split("__", 1)[-1]
        for f in ordered:
            if name == f or name.startswith(f + "_"):
                return f
        return name

    imp = pd.Series(est.feature_importances_, index=names)
    return imp.groupby(origin).sum().sort_values(ascending=False)


# ---- input sanity checks ----
def validate_inputs(area, bhk, flat_type, property_type, locality=None, company=None):
    """Returns (errors, warnings). Errors block the estimate; warnings mark it unreliable."""
    errors, warns = [], []
    ft, pt = flat_type.lower(), property_type.lower()
    is_plot = "plot" in ft or "plot" in pt

    if is_plot and bhk > 0:
        errors.append(f"A plot has no rooms, but BHK count is {bhk}. Set BHK to 0.")
    m = re.search(r"(\d+)\s*bhk", pt)
    if m and int(m.group(1)) != bhk:
        errors.append(f"Property type says {m.group(1)} BHK but BHK count is {bhk}.")

    if not is_plot and bhk == 0:
        warns.append("BHK count is 0 for a built property (studio or data gap?).")
    if bhk > 0:
        per_bhk = area / bhk
        if per_bhk < 250:
            warns.append(f"Only ~{per_bhk:,.0f} sq. ft. per BHK, which is unusually small.")
        elif per_bhk > 3500:
            warns.append(f"~{per_bhk:,.0f} sq. ft. per BHK is unusually large.")

    if df is not None:
        chosen = {"Flat Type": flat_type, "Property Type": property_type,
                  "Locality": locality, "Company Name": company}
        for a_col, b_col in (("Flat Type", "Property Type"), ("Locality", "Flat Type"),
                             ("Locality", "Company Name"), ("Company Name", "Flat Type")):
            if (a_col in df.columns and b_col in df.columns
                    and chosen.get(a_col) and chosen.get(b_col)):
                together = (df[a_col] == chosen[a_col]) & (df[b_col] == chosen[b_col])
                if not together.any():
                    warns.append(
    f"The combination of {a_col} '{chosen[a_col]}' and "
    f"{b_col} '{chosen[b_col]}' was not found in the training dataset. "
    "The model can still generate an estimate, but this combination has "
    "limited direct data support. Interpret the result cautiously."
)
        if {"Area", "Flat Type"} <= set(df.columns):
            sub = df.loc[df["Flat Type"] == flat_type, "Area"].dropna()
            if len(sub) >= 30:
                lo, hi = sub.quantile([0.01, 0.99])
                if not lo <= area <= hi:
                    warns.append(f"Area is outside the typical {flat_type} range in the data "
                                 f"({lo:,.0f} to {hi:,.0f} sq. ft.), so the model is extrapolating.")
        if locality and "Locality" in df.columns:
            n_loc = (df["Locality"] == locality).sum()
            if n_loc < 5:
                warns.append(f"Only {n_loc} listings exist for {locality}; its estimate is weakly supported.")
    return errors, warns


def comparable_listings(locality, flat_type, bhk, area=None):
    """Return comparable listings using progressively broader filters."""
    if df is None or not {"Price", "Locality", "Flat Type", "BHK_Count"} <= set(df.columns):
        return None, None

    base = df[df["Price"].gt(0)].copy()

    # Normalize fields so comparisons are consistent.
    base["_locality"] = base["Locality"].astype("string").str.strip()
    base["_flat_type"] = base["Flat Type"].astype("string").str.strip()
    base["_bhk"] = pd.to_numeric(base["BHK_Count"], errors="coerce")
    base["_area"] = (
        pd.to_numeric(base["Area"], errors="coerce")
        if "Area" in base.columns
        else float("nan")
    )

    locality_value = str(locality).strip()
    flat_value = str(flat_type).strip()

    same_locality = base["_locality"].eq(locality_value)
    same_type = base["_flat_type"].eq(flat_value)
    same_bhk = base["_bhk"].eq(float(bhk))

    # Try the most relevant group first.
    tiers = [
        ("same locality, flat type and BHK",
         same_locality & same_type & same_bhk),
        ("same flat type and BHK",
         same_type & same_bhk),
        ("same locality and flat type",
         same_locality & same_type),
        ("same flat type",
         same_type),
    ]

    for label, mask in tiers:
        candidates = base.loc[mask].copy()

        if area is not None and "_area" in candidates.columns:
            # Prefer listings within +/-30% of the requested area.
            area_num = pd.to_numeric(pd.Series([area]), errors="coerce").iloc[0]

            if pd.notna(area_num) and area_num > 0:
                area_matches = candidates[
                    candidates["_area"].between(area_num * 0.70, area_num * 1.30)
                ]

                # Use the narrow area band only if it has enough data.
                if len(area_matches) >= 8:
                    candidates = area_matches
                    label += ", similar area"

        if len(candidates) >= 8:
            return candidates.drop(
                columns=["_locality", "_flat_type", "_bhk", "_area"],
                errors="ignore"
            ), label

    return None, None


def smart_defaults() -> dict:
    """Start from a REAL listing: the one nearest the median area inside the most common
    (Flat Type, Property Type, BHK) combination, so every default is jointly valid."""
    d = {"Flat Type": None, "Property Type": None, "Locality": None, "Status": None,
         "RERA Approval": None, "Company Name": None, "BHK_Count": 3, "Area": 1500}
    if df is None:
        return d
    try:
        need = [c for c in FEATURES if c in df.columns]
        base = df.dropna(subset=need)
        if base.empty:
            return d
        cols = [c for c in ("Flat Type", "Property Type", "BHK_Count") if c in base.columns]
        sub = base
        if cols:
            top = base.groupby(cols).size().sort_values(ascending=False).index[0]
            top = top if isinstance(top, tuple) else (top,)
            for c, v in zip(cols, top):
                sub = sub[sub[c] == v]
        if "Area" in sub.columns:
            pos = (sub["Area"] - sub["Area"].median()).abs().to_numpy().argmin()
            row = sub.iloc[pos]
        else:
            row = sub.iloc[0]
        for c in ("Flat Type", "Property Type", "Locality", "Status", "RERA Approval", "Company Name"):
            if c in row.index:
                d[c] = str(row[c])
        if "BHK_Count" in row.index:
            d["BHK_Count"] = int(min(10, max(0, row["BHK_Count"])))
        if "Area" in row.index:
            d["Area"] = int(min(100000, max(100, round(float(row["Area"]) / 100) * 100)))
    except Exception:
        pass
    return d


def empirical_band(pred: float):
    """Approximate test-error range; not a confidence interval."""
    try:
        pred = float(pred)
    except (TypeError, ValueError):
        return None

    if (
        not np.isfinite(pred)
        or pred <= 0
        or not metrics
        or metrics_status != "verified"
    ):
        return None

    yt = metrics.get("y_true")
    yp = metrics.get("y_pred")

    if yt is not None and yp is not None:
        try:
            yt = np.asarray(yt, dtype=float).ravel()
            yp = np.asarray(yp, dtype=float).ravel()

            if len(yt) == len(yp):
                ok = (
                    np.isfinite(yt) & np.isfinite(yp)
                    & (yt > 0) & (yp > 0)
                )
                yt, yp = yt[ok], yp[ok]

                if len(yt) >= 50:
                    ratio = yt / yp

                    if len(yt) >= 150:
                        k = max(60, len(yt) // 4)
                        near = np.argsort(
                            np.abs(np.log(yp) - np.log(pred))
                        )[:k]
                        lo, hi = np.quantile(ratio[near], [0.10, 0.90])
                        label = f"{k} held-out test rows at a similar predicted price"
                    else:
                        lo, hi = np.quantile(ratio, [0.10, 0.90])
                        label = f"all {len(yt)} usable held-out test rows"

                    if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
                        return (
                            max(0.0, pred * float(lo)),
                            max(0.0, pred * float(hi)),
                            label,
                        )
        except (TypeError, ValueError, OverflowError):
            pass

    # Fallback: show the MAE-based guide only when MAE is valid.
    try:
        mae = float(metrics.get("mae"))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(mae) or mae < 0:
        return None

    return (
        max(0.0, pred - mae),
        pred + mae,
        "± held-out test MAE (rough guide)",
    )


def support_count(
    locality,
    flat_type,
    bhk=None,
    property_type=None,
    area=None,
) -> int:
    required = {"Locality", "Flat Type"}

    if df is None or not required.issubset(df.columns):
        return 0

    loc = df["Locality"].astype("string").str.strip()
    ft = df["Flat Type"].astype("string").str.strip()

    mask = (
        loc.eq(str(locality).strip())
        & ft.eq(str(flat_type).strip())
    )

    if bhk is not None and "BHK_Count" in df.columns:
        bhk_values = pd.to_numeric(df["BHK_Count"], errors="coerce")
        mask &= bhk_values.eq(float(bhk))

    if property_type is not None and "Property Type" in df.columns:
        pt = df["Property Type"].astype("string").str.strip()
        mask &= pt.eq(str(property_type).strip())

    if "Price" in df.columns:
        prices = pd.to_numeric(df["Price"], errors="coerce")
        mask &= prices.gt(0)

    # Count only listings within +/-30% of the requested area.
    if area is not None and "Area" in df.columns:
        areas = pd.to_numeric(df["Area"], errors="coerce")
        area = float(area)

        if np.isfinite(area) and area > 0:
            mask &= areas.between(area * 0.70, area * 1.30)

    return int(mask.fillna(False).sum())


def locality_sweep(inp: dict, top_n: int = 12):
    """Same property predicted in the most-listed localities."""
    if df is None or "Locality" not in df.columns:
        return None
    locs = df["Locality"].value_counts().head(top_n).index.tolist()
    if inp["Locality"] not in locs:
        locs.append(inp["Locality"])
    rows = pd.DataFrame([{**inp, "Locality": loc} for loc in locs], columns=FEATURES)
    rows["Predicted price"] = point_predict(model, rows)
    return rows


@st.cache_data(show_spinner=False)
def scale_check(_model, model_sha, data_sha, log_target):
    """Does the model predict prices on the same scale as the dataset? Catches LOG_TARGET mistakes."""
    if df is None or "Price" not in df.columns:
        return "skip", "No dataset available to compare predictions against."
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        return "fail", f"Dataset is missing model feature columns: {missing}"
    sample = df.dropna(subset=FEATURES + ["Price"])
    sample = sample[sample["Price"] > 0]
    if sample.empty:
        return "skip", "No complete dataset rows to test with."
    sample = sample.sample(min(200, len(sample)), random_state=0)
    try:
        preds = point_predict(_model, sample[FEATURES])
    except Exception as exc:
        return "fail", f"The model could not predict on dataset rows: {exc}"
    ratio = float(np.median(preds) / np.median(sample["Price"]))
    if not np.isfinite(ratio) or not 0.05 < ratio < 20:
        return "fail", (f"Predictions are about {ratio:.3g}x the dataset prices. "
                        "Check LOG_TARGET at the top of app.py and that the right model file is loaded.")
    return "ok", f"Median prediction is {ratio:.2f}x the median dataset price (scale looks right)."


def category_coverage():
    """Compare dataset categories with the model's fitted encoder."""
    if df is None or model is None:
        return None

    try:
        pre = model.named_steps["preprocessor"]
        cat_pipeline = pre.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["encoder"]
        imputer = cat_pipeline.named_steps["imputer"]

        if not hasattr(encoder, "categories_"):
            return None

        categorical = [
            "Status",
            "Property Type",
            "Locality",
            "RERA Approval",
            "Flat Type",
            "Company Name",
        ]

        # Get feature names from the fitted ColumnTransformer
        cat_features = None

        for name, transformer, columns in pre.transformers_:
            if name == "cat":
                cat_features = list(columns)
                break

        if cat_features is None:
            return None

        out = {}

        for feat, cats in zip(cat_features, encoder.categories_):
            if feat not in categorical or feat not in df.columns:
                continue

            values = df[feat].fillna(
                imputer.statistics_[cat_features.index(feat)]
            ).astype(str).str.strip()

            known = {str(c).strip() for c in cats}

            if not values.empty:
                out[feat] = float(values.isin(known).mean())

        return out or None

    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


def health_checks() -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    add = lambda state, label, detail: checks.append((state, label, detail))

    add("ok", "Model loaded",
        f"{type(final_estimator(model)).__name__} · fingerprint {MODEL_SHA[:10]}…")
    if df is None:
        add("warn", "Dataset", "Not loaded, so analytics and comparisons are unavailable.")
    else:
        add("ok", "Dataset loaded", f"{DATA_PATH.name} · {len(df):,} rows")
        missing = [c for c in FEATURES if c not in df.columns]
        add("fail" if missing else "ok", "Feature columns present",
            f"Missing: {missing}" if missing else "All 8 model features exist in the CSV.")
    if len(EXISTING_DATA) > 1 and not DATA_FILE:
        add("warn", "Multiple CSV files found",
            ", ".join(p.name for p in EXISTING_DATA) + ". Keep only the training file or set DATA_FILE.")

    state, detail = scale_check(model, MODEL_SHA, DATA_SHA, LOG_TARGET)
    add({"ok": "ok", "skip": "warn", "fail": "fail"}[state], "Prediction scale", detail)

    cov = category_coverage()
    if cov is None:
        add("warn", "Category coverage", "Could not read the fitted encoder's categories (custom pipeline?).")
    else:
        worst_f, worst_v = min(cov.items(), key=lambda kv: kv[1])
        if worst_v < 0.90:
            add("fail", "Category coverage",
                f"Only {worst_v:.0%} of '{worst_f}' values are known to the model's encoder. "
                "Check spelling, spaces or case differences between training data and this CSV.")
        else:
            add("ok", "Category coverage",
                f"At least {worst_v:.0%} of every category column matches the model's encoder.")

    status_map = {
        "verified": ("ok", "Metrics file", "Linked to this exact model file."),
        "unverified": ("warn", "Metrics file", "Has no model fingerprint. Regenerate with evaluate_model.py."),
        "mismatch": ("fail", "Metrics file", "Belongs to a different model file and is hidden."),
        "missing": ("warn", "Metrics file", "Not found. Run evaluate_model.py to create it."),
    }
    add(*status_map[metrics_status])
    for note in metrics_notes:
        add("warn", "Environment / data", note)
    return checks


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:25px;font-weight:700;font-family:Space Grotesk'>"
        "◈ Estate<span style='color:#48E0C1'>IQ</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("PROPERTY INTELLIGENCE PLATFORM")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Price Predictor", "Market Analytics", "Model Performance"],
    )
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Trained model loaded")
    st.caption(f"Algorithm: {type(final_estimator(model)).__name__}")
    sc_state, sc_msg = scale_check(model, MODEL_SHA, DATA_SHA, LOG_TARGET)
    if sc_state == "fail":
        st.error("Model/data check failed. See Model Performance → System health.")
    if DATA_PATH is not None:
        st.caption(f"Dataset: {DATA_PATH.name}")
    st.caption("Region: Gurgaon, Haryana")
    if metrics and metrics.get("r2") is not None:
        tag = "verified" if metrics_status == "verified" else "unverified"
        st.caption(f"Test R²: {metrics['r2']:.3f} ({tag})")
    elif metrics_status == "mismatch":
        st.caption("⚠ Metrics do not match this model file")
    st.divider()
    st.caption("EstateIQ • Academic ML Project")

if data_error:
    st.warning(f"Dataset analytics could not load: {data_error}")
if len(EXISTING_DATA) > 1 and not DATA_FILE:
    st.warning("More than one dataset CSV was found (" + ", ".join(p.name for p in EXISTING_DATA)
               + f"). Using **{DATA_PATH.name}**. Keep only the file the model was trained on, "
               "or set `DATA_FILE` at the top of app.py.")


# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------
if page == "Overview":
    st.markdown(
        """<div class="hero">
          <div class="eyebrow">REAL ESTATE • MACHINE LEARNING</div>
          <h1>Property intelligence,<br>made clearer.</h1>
          <p>Estimate Gurgaon property values with a trained Random Forest, see how
          each estimate compares with similar listings, and explore the market data
          behind it.</p></div>""",
        unsafe_allow_html=True,
    )

    if df is None:
        st.info("Dataset not found. Prediction still works; add the CSV in the "
                "data folder to enable analytics.")
    else:
        section("Workspace overview")
        cols = st.columns(4)
        median_price = df["Price"].median() if "Price" in df.columns else None
        with cols[0]:
            glass_card("DATASET RECORDS", f"{len(df):,}", "Rows in the CSV")
        with cols[1]:
            glass_card("MEDIAN LISTED PRICE",
                       money_inr(median_price) if pd.notna(median_price) else "N/A",
                       "Dataset median, not live market data")
        with cols[2]:
            glass_card("MEDIAN PRICE / SQ. FT.",
                       f"₹{df['Price_per_sqft'].median():,.0f}"
                       if "Price_per_sqft" in df.columns else "N/A",
                       "Fairer than raw price across sizes")
        with cols[3]:
            glass_card("LOCALITIES",
                       f"{df['Locality'].nunique():,}" if "Locality" in df.columns else "N/A",
                       "Unique locality values")

        if "Price" in df.columns:
            section("Dataset snapshot")
            show(price_hist(df["Price"], 45, log=True,
                            title="Price distribution (log scale)"))
        if {"Locality", "Price_per_sqft"} <= set(df.columns):
            loc_stats = (df.dropna(subset=["Price_per_sqft"]).groupby("Locality")
                         .agg(psf=("Price_per_sqft", "median"), n=("Price", "size"))
                         .query("n >= 10"))
            if len(loc_stats) >= 2:
                section("Market highlights")
                h1, h2, h3 = st.columns(3)
                top_loc, low_loc = loc_stats["psf"].idxmax(), loc_stats["psf"].idxmin()
                busy = df["Locality"].value_counts().idxmax()
                with h1:
                    glass_card("HIGHEST PRICE / SQ. FT.", top_loc,
                               f"₹{loc_stats.loc[top_loc, 'psf']:,.0f} median (10+ listings)")
                with h2:
                    glass_card("MOST AFFORDABLE / SQ. FT.", low_loc,
                               f"₹{loc_stats.loc[low_loc, 'psf']:,.0f} median (10+ listings)")
                with h3:
                    glass_card("MOST LISTINGS", busy,
                               f"{(df['Locality'] == busy).sum():,} properties in the data")
        st.caption("Dataset summaries describe the loaded CSV. They are not live "
                   "listings or verified market benchmarks.")

    section("How it works")
    a, b, c = st.columns(3)
    with a:
        glass_card("01 · INPUT", "Property details", "Area, locality, configuration, status")
    with b:
        glass_card("02 · MODEL", "Random Forest", "Trained pipeline plus per-tree spread")
    with c:
        glass_card("03 · OUTPUT", "Estimate + context", "Range and comparison with similar listings")


# --------------------------------------------------
# 7. PRICE PREDICTOR
# --------------------------------------------------
elif page == "Price Predictor":
    page_header(
        "Price Predictor",
        "Describe a property to get an estimate, an empirical error range and a market comparison."
    )

    status_options = options_from_data("Status", ["Ready to move", "Under construction"])
    property_options = options_from_data("Property Type", [])
    locality_options = options_from_data("Locality", ["Sector 65", "Sector 62", "Golf Course Road"])
    rera_options = options_from_data(
        "RERA Approval",
        ["Approved by RERA", "Not approved by RERA", "Not Applicable"])
    flat_options = options_from_data(
        "Flat Type", ["Apartment", "Plot", "Floor", "Villa", "House", "Penthouse"])
    company_options = options_from_data("Company Name", ["M3M", "DLF", "Godrej", "Unspecified"])

    defaults = smart_defaults()

    if sc_state == "fail":
        st.error(f"**Model/data check failed:** {sc_msg}")

    if df is not None:
        st.caption(
            "Defaults come from the most common real combination in your dataset, "
            "so the starting point is always a valid property."
        )

    st.markdown("#### 📐 Size & configuration")
    c1, c2, c3 = st.columns(3)

    with c1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            max_value=100000,
            value=int(defaults["Area"]),
            step=100,
            key="area_input",
        )

    with c2:
        bhk = st.number_input(
            "BHK count",
            min_value=0,
            max_value=10,
            value=int(defaults["BHK_Count"]),
            step=1,
            key="bhk_count",
            on_change=reset_invalid_property_type,
        )

    with c3:
        flat_type = st.selectbox(
            "Flat type",
            flat_options,
            index=default_index(flat_options, defaults["Flat Type"]),
            key="flat_type",
            on_change=reset_invalid_property_type,
        )

    # Filter property types according to selected BHK and flat type
    filtered_property_options = property_types_for_bhk_and_flat(df, bhk, flat_type)

    # Keep the selected property type valid for the selected BHK
    if filtered_property_options:
        current_property_type = st.session_state.get("property_type")
        if current_property_type not in filtered_property_options:
            if defaults["Property Type"] in filtered_property_options:
                st.session_state["property_type"] = defaults["Property Type"]
            else:
                st.session_state["property_type"] = filtered_property_options[0]
    else:
        st.session_state.pop("property_type", None)

    st.markdown("#### 📍 Location & property details")
    c4, c5 = st.columns(2)

    with c4:
        locality = st.selectbox(
            "Locality",
            locality_options,
            index=default_index(locality_options, defaults["Locality"]),
            key="locality"
        )

        status = st.selectbox(
            "Status",
            status_options,
            index=default_index(status_options, defaults["Status"]),
            key="status"
        )

        if filtered_property_options:
            property_type = st.selectbox(
                "Property type",
                filtered_property_options,
                key="property_type"
            )
        else:
            property_type = None
            st.warning(
                "No matching property type found for this BHK in the dataset. "
                "Try another BHK count."
            )

    with c5:
        company = st.selectbox(
            "Builder / company",
            company_options,
            index=default_index(company_options, defaults["Company Name"]),
            key="company"
        )

        rera = st.selectbox(
            "RERA approval",
            rera_options,
            index=default_index(rera_options, defaults["RERA Approval"]),
            key="rera"
        )

    submitted = st.button(
        "✦  Estimate property value",
        disabled=not bool(filtered_property_options),
        **STRETCH
    )

    if submitted:
        inputs = {
            "Area": area, "Status": status, "Property Type": property_type,
            "Locality": locality, "RERA Approval": rera, "BHK_Count": bhk,
            "Flat Type": flat_type, "Company Name": company,
        }
        errors, warns = validate_inputs(area, bhk, flat_type, property_type, locality, company)
        if errors:
            st.session_state.pop("result", None)
            st.error("**These inputs contradict each other, so no estimate was made:**\n\n"
                     + "\n".join(f"- {e}" for e in errors))
        else:
            X = pd.DataFrame([inputs], columns=FEATURES)
            try:
                pred = float(point_predict(model, X)[0])
                if not np.isfinite(pred) or pred <= 0:
                    st.session_state.pop("result", None)
                    st.error("The model returned an invalid estimate. Review the inputs.")
                else:
                    rng, rng_reason = predict_range(model, X)
                    st.session_state["result"] = {
                        "inputs": inputs, "pred": pred, "range": rng,
                        "range_reason": rng_reason, "issues": warns,
                    }
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Prediction failed: {exc}")
                st.info("Confirm the eight feature names and types match the training pipeline.")

    res = st.session_state.get("result")
    if res:
        inp, pred, rng, issues = res["inputs"], res["pred"], res["range"], res["issues"]
        area, bhk = inp["Area"], inp["BHK_Count"]
        psf = pred / area
        band = empirical_band(pred)

        section("Valuation result")
        if issues:
            st.warning("**Estimate may be unreliable:**\n\n" + "\n".join(f"- {i}" for i in issues))

        left, right = st.columns([1.5, 1])
        with left:
            badge = ('<span class="badge badge-warn">LOW CONFIDENCE INPUTS</span>'
                     if issues else '<span class="badge badge-ok">CONSISTENT INPUTS</span>')
            lines = [f"<p>Point estimate: {money_full(pred)}</p>"]
            if band:
                lines.append(f"<p>Empirical test-based range (approx.): <b>{money_inr(band[0])} – {money_inr(band[1])}</b> "
                             f"<span style='color:#91A0B7'>({band[2]})</span></p>")
            if rng:
                lines.append(f"<p>Tree-spread range: <b>{money_inr(rng[0])} – {money_inr(rng[1])}</b> "
                             "<span style='color:#91A0B7'>(model disagreement only)</span></p>")
            st.markdown(
                f"""<div class="hero" style="margin-top:0">
                  <div class="eyebrow">MODEL ESTIMATE &nbsp; {badge}</div>
                  <div style="font-size:clamp(34px,5vw,50px);font-weight:700;
                              font-family:Space Grotesk;margin:12px 0;color:#48E0C1">
                    {money_inr(pred)}</div>{''.join(lines)}</div>""",
                unsafe_allow_html=True,
            )
            notes = []
            if band is None:
                notes.append("Empirical range unavailable: run `evaluate_model.py` to create a verified `model_metrics.json` "
                             "(see Model Performance).")
            elif metrics_status == "unverified":
                notes.append("The band uses a metrics file that is not linked to this model file.")
            if rng is None:
                notes.append(f"Tree-spread range unavailable: {res.get('range_reason')}")
            notes.append("The tree-spread range is usually much narrower than real error. The empirical "
                         "range is approximate and has no guaranteed coverage for any single property. "
                         "Neither is a valuation or a confidence interval.")
            st.caption(" ".join(notes))
        with right:
            st.markdown('<div class="section-title" style="margin-top:0">Property summary</div>',
                        unsafe_allow_html=True)
            st.write(f"**Area:** {area:,} sq. ft.")
            st.write(f"**Configuration:** {bhk} BHK")
            st.write(f"**Locality:** {inp['Locality']}")
            st.write(f"**Type:** {inp['Flat Type']} · {inp['Property Type']}")
            st.write(f"**Status:** {inp['Status']}")

        # ---- context vs dataset ----
        comps, comp_label = comparable_listings(
            inp["Locality"],
            inp["Flat Type"],
            bhk,
            area=area,
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estimated price / sq. ft.", f"₹{psf:,.0f}")
        n_support = support_count(
            inp["Locality"],
            inp["Flat Type"],
            bhk=bhk,
            property_type=inp["Property Type"],
            area=area,
        )

        level = "High" if n_support >= 30 else "Medium" if n_support >= 8 else "Low"
        k4.metric("Data support", level, f"{n_support:,} similar listings", delta_color="off")
        if comps is not None:
            med = comps["Price"].median()
            delta = pred / med - 1
            k2.metric("Similar listings (median)", money_inr(med), f"{delta:+.0%} vs estimate",
                      delta_color="off")
            if "Price_per_sqft" in comps.columns and comps["Price_per_sqft"].notna().any():
                k3.metric("Similar price / sq. ft.", f"₹{comps['Price_per_sqft'].median():,.0f}")
            verdict = ("in line with" if abs(delta) < 0.10 else "above" if delta > 0 else "below")
            section("How this compares")
            st.write(f"The estimate is **{verdict}** the median of {len(comps):,} "
                     f"comparable listings ({comp_label}).")
            fig = price_hist(comps["Price"], 25, log=False,
                             title=f"Comparable listings ({comp_label})")
            fig.add_vline(x=pred, line_color="#FAB005", line_width=3,
                          annotation_text="Your estimate", annotation_position="top")
            if band:
                fig.add_vrect(x0=band[0], x1=band[1], fillcolor="#FAB005", opacity=0.10,
                              line_width=0)
            show(fig)

            cols_show = [c for c in ("Locality", "Flat Type", "Property Type", "BHK_Count",
                                     "Area", "Price", "Price_per_sqft", "Company Name")
                         if c in comps.columns]
            if "Area" in comps.columns:
                sim = comps.assign(_gap=(comps["Area"] - area).abs()).sort_values("_gap")
            else:
                sim = comps
            sim = sim.head(8)[cols_show].copy()
            sim["Price"] = sim["Price"].map(money_inr)
            if "Price_per_sqft" in sim.columns:
                sim["Price_per_sqft"] = sim["Price_per_sqft"].map(
                    lambda v: f"₹{v:,.0f}" if pd.notna(v) else "N/A")
            st.markdown("**Closest real listings by area**")
            st.dataframe(sim, hide_index=True, **STRETCH)
        else:
            k2.metric("Area", f"{area:,} sq. ft.")
            k3.metric("BHK", str(bhk))
            st.caption("Not enough comparable listings in the dataset for a market comparison.")

        # ---- what-if: price vs area ----
        try:
            factors = np.linspace(0.5, 2.0, 25)
            sweep = pd.DataFrame([{**inp, "Area": max(100, int(area * f))} for f in factors],
                                 columns=FEATURES)
            sweep["Predicted price"] = point_predict(model, sweep)
            fig = px.line(sweep, x="Area", y="Predicted price", labels={"Area": "Area (sq. ft.)"})
            fig.update_traces(line_color=ACCENT)
            fig.add_trace(go.Scatter(x=[area], y=[pred], mode="markers",
                                     marker=dict(size=13, color="#FAB005"), name="Your input"))
            section("What-if: how price responds to area")
            show(style_fig(fig, title="Model-predicted price vs area (other inputs fixed)"))
        except Exception:
            pass

        # ---- what-if: same property, other localities ----
        try:
            loc_rows = locality_sweep(inp)
            if loc_rows is not None:
                loc_rows = loc_rows.sort_values("Predicted price")
                loc_rows["Selected"] = np.where(loc_rows["Locality"] == inp["Locality"],
                                                "Your locality", "Other")
                fig = px.bar(loc_rows, x="Predicted price", y="Locality", orientation="h",
                             color="Selected",
                             color_discrete_map={"Your locality": "#FAB005", "Other": ACCENT_2},
                             labels={"Predicted price": "Model estimate (₹)"})
                section("What-if: same property, other localities")
                show(style_fig(fig, height=max(360, 30 * len(loc_rows) + 120),
                               title="Estimate for this exact configuration in the most-listed localities"))
        except Exception:
            pass

        summary = pd.DataFrame([{
            "Estimated Price INR": round(pred, 2),
            "Empirical Range Low INR": round(band[0], 2) if band else None,
            "Empirical Range High INR": round(band[1], 2) if band else None,
            "Tree Range Low INR": round(rng[0], 2) if rng else None,
            "Tree Range High INR": round(rng[1], 2) if rng else None,
            "Estimated Price per Sq Ft INR": round(psf, 2),
            **inp,
        }])
        st.download_button("⬇ Download prediction summary (CSV)",
                           data=summary.to_csv(index=False).encode("utf-8"),
                           file_name="property_prediction.csv", mime="text/csv", **STRETCH)


# --------------------------------------------------
# 8. MARKET ANALYTICS
# --------------------------------------------------
elif page == "Market Analytics":
    page_header("Market Analytics", "Filter the dataset and explore price patterns.")

    if df is None or "Price" not in df.columns:
        st.warning("Analytics need the project CSV with a Price column. "
                   "Check the data folder.")
    else:
        data = df[df["Price"] > 0].copy()

        with st.expander("Filters", expanded=True):
            f1, f2, f3 = st.columns(3)
            sel_loc = f1.multiselect("Locality", sorted(data["Locality"].dropna().unique())
                                     if "Locality" in data.columns else [])
            sel_type = f2.multiselect("Flat type", sorted(data["Flat Type"].dropna().unique())
                                      if "Flat Type" in data.columns else [])
            sel_status = f3.multiselect("Status", sorted(data["Status"].dropna().unique())
                                        if "Status" in data.columns else [])
            g1, g2, g3 = st.columns(3)
            if "BHK_Count" in data.columns and data["BHK_Count"].notna().any():
                lo, hi = int(data["BHK_Count"].min()), int(data["BHK_Count"].max())
                bhk_range = g1.slider("BHK range", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
            else:
                bhk_range = None
            clip = g2.toggle("Clip outliers at 99th percentile", value=True)
            log_scale = g3.toggle("Log price axis", value=True)

        if sel_loc:
            data = data[data["Locality"].isin(sel_loc)]
        if sel_type:
            data = data[data["Flat Type"].isin(sel_type)]
        if sel_status:
            data = data[data["Status"].isin(sel_status)]
        if bhk_range:
            data = data[data["BHK_Count"].between(*bhk_range)]
        if clip and len(data) > 20:
            data = data[data["Price"] <= data["Price"].quantile(0.99)]

        if data.empty:
            st.warning("No listings match these filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Listings", f"{len(data):,}")
            m2.metric("Median price", money_inr(data["Price"].median()))
            if "Price_per_sqft" in data.columns:
                m3.metric("Median price / sq. ft.", f"₹{data['Price_per_sqft'].median():,.0f}")
            m4.metric("Price spread (P25 to P75)",
                      f"{money_inr(data['Price'].quantile(.25))} – "
                      f"{money_inr(data['Price'].quantile(.75))}")

            t1, t2, t3, t4, t5 = st.tabs(
                ["Distribution", "Price vs Area", "Localities", "By BHK", "Data"])

            with t1:
                show(price_hist(data["Price"], 50, log_scale, "Price distribution"))

            with t2:
                if {"Area", "Price"} <= set(data.columns):
                    sc = data.dropna(subset=["Area"])
                    sc = sc[sc["Area"] > 0]
                    if len(sc) > 4000:
                        sc = sc.sample(4000, random_state=42)
                    fig = px.scatter(
                        sc, x="Area", y="Price", opacity=0.55,
                        color="Flat Type" if "Flat Type" in sc.columns else None,
                        hover_data=[c for c in ("Locality", "BHK_Count") if c in sc.columns],
                        log_x=log_scale, log_y=log_scale,
                        labels={"Area": "Area (sq. ft.)", "Price": "Listed price (₹)"},
                    )
                    show(style_fig(fig, height=520,
                                   title="Price vs area (sampled to 4,000 points)"))

            with t3:
                if "Locality" in data.columns and "Price_per_sqft" in data.columns:
                    min_n = st.slider("Minimum listings per locality", 3, 30, 5)
                    top_n = st.slider("Localities to show", 5, 30, 15)
                    loc = (data.groupby("Locality")
                           .agg(Median_PSF=("Price_per_sqft", "median"),
                                Median_Price=("Price", "median"),
                                Listings=("Price", "size"))
                           .query("Listings >= @min_n")
                           .sort_values("Median_PSF", ascending=False)
                           .head(top_n).reset_index())
                    if loc.empty:
                        st.info("No locality has enough listings for these filters.")
                    else:
                        fig = px.bar(loc.sort_values("Median_PSF"), x="Median_PSF", y="Locality",
                                     orientation="h", hover_data=["Listings", "Median_Price"],
                                     labels={"Median_PSF": "Median price per sq. ft. (₹)"})
                        fig.update_traces(marker_color=ACCENT_2)
                        show(style_fig(fig, height=max(380, 28 * len(loc) + 120),
                                       title="Median price per sq. ft. by locality"))
                        st.caption("Price per sq. ft. is compared instead of raw price, "
                                   "so large and small homes are on equal footing.")

            with t4:
                if "BHK_Count" in data.columns:
                    bx = data.dropna(subset=["BHK_Count"]).copy()
                    bx["BHK"] = bx["BHK_Count"].astype(int).astype(str)
                    fig = px.box(bx.sort_values("BHK_Count"), x="BHK", y="Price",
                                 log_y=log_scale, points=False,
                                 labels={"Price": "Listed price (₹)"})
                    fig.update_traces(marker_color=ACCENT, line_color=ACCENT)
                    show(style_fig(fig, title="Price by BHK count"))

            with t5:
                st.dataframe(data.head(1000), **STRETCH)
                st.download_button("⬇ Download filtered data (CSV)",
                                   data=data.to_csv(index=False).encode("utf-8"),
                                   file_name="filtered_listings.csv", mime="text/csv")

            st.caption("Charts describe your dataset, not current market prices.")


# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    page_header(
        "Model Performance",
        "How good is the model, and what drives its predictions?"
    )

    est = final_estimator(model)

    # Metrics verification status
    if metrics_status == "verified":
        extra = (
            " Reproduction check against your notebook numbers passed."
            if metrics and metrics.get("reproduced_expected")
            else ""
        )
        st.success(
            f"Metrics verified: they belong to this exact model file "
            f"(fingerprint {MODEL_SHA[:10]}…)." + extra
        )

    elif metrics_status == "unverified":
        st.warning(
            "Metrics file has no model fingerprint, so it cannot be proven "
            "to belong to this model. Regenerate it with the snippet below."
        )

    elif metrics_status == "mismatch":
        st.error(
            "`model_metrics.json` was created for a DIFFERENT model file, "
            "so its numbers are hidden. Regenerate it with the snippet below."
        )

    else:
        st.info(f"No metrics file found at `models/{METRICS_PATH.name}`.")

    for note in metrics_notes:
        st.warning(note)

    for warning in (metrics or {}).get("warnings", []):
        st.warning(warning)

    # System health checks
    checks = health_checks()
    bad = [check for check in checks if check[0] != "ok"]

    with st.expander(
        f"System health checks ({len(checks) - len(bad)}/{len(checks)} passed)",
        expanded=bool(bad)
    ):
        icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}

        for state, label, detail in checks:
            st.markdown(
                f"{icon.get(state, 'ℹ️')} **{label}**: {detail}"
            )

    # Model details
    section("Model details")
    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Algorithm", type(est).__name__.replace("Regressor", ""))
    d2.metric("Trees", str(getattr(est, "n_estimators", "N/A")))
    d3.metric("Max depth", str(getattr(est, "max_depth", None) or "None"))
    d4.metric("scikit-learn", SKLEARN_VERSION or "N/A")

    # Accuracy metrics
    section("Accuracy on held-out data")

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **R²** is the share of price variation across properties the model explains. "
            "An R² of 0.80 does not mean each estimate is 80% accurate.\n"
            "- **MAE** is the average size of a miss in rupees; every miss counts equally.\n"
            "- **RMSE** squares misses first, so a few badly mispriced, expensive listings push it "
            "well above MAE."
        )

        if metrics:
            try:
                mae_value = float(metrics.get("mae"))
                rmse_value = float(metrics.get("rmse"))

                if (
                    np.isfinite(mae_value)
                    and np.isfinite(rmse_value)
                    and mae_value > 0
                    and rmse_value >= 0
                ):
                    ratio_ = rmse_value / mae_value
                    st.markdown(
                        f"Here RMSE is **{ratio_:.1f}×** MAE."
                        + (
                            " Large misses on a few expensive listings dominate the error."
                            if ratio_ > 1.5 else ""
                        )
                    )
            except (TypeError, ValueError):
                pass

    if metrics:
        cols = st.columns(4)

        r2 = metrics.get("r2")
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")

        with cols[0]:
            glass_card(
                "R² SCORE",
                f"{r2:.3f}" if r2 is not None else "N/A",
                "Share of price variance explained"
            )

        with cols[1]:
            glass_card(
                "MEAN ABS. ERROR",
                money_inr(mae) if mae is not None else "N/A",
                "Typical miss per prediction"
            )

        with cols[2]:
            glass_card(
                "RMSE",
                money_inr(rmse) if rmse is not None else "N/A",
                "Penalises large misses more"
            )

        with cols[3]:
            n_test = metrics.get(
                "test_rows",
                metrics.get("evaluation_rows", metrics.get("n_test"))
            )
            n_train = metrics.get("train_rows", metrics.get("n_train"))

            glass_card(
                "TEST ROWS",
                f"{n_test:,}" if n_test is not None else "N/A",
                f"Train rows: {n_train:,}" if n_train is not None else "Train rows: N/A"
            )

        # Error compared with median listing price
        if mae is not None and df is not None and "Price" in df.columns:
            try:
                median_price = pd.to_numeric(
                    df["Price"], errors="coerce"
                ).median()

                if (
                    pd.notna(median_price)
                    and median_price > 0
                    and np.isfinite(float(mae))
                ):
                    st.caption(
                        f"Typical error is about {float(mae) / median_price:.0%} "
                        "of the dataset's median listed price."
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Actual vs predicted plot
        yt = metrics.get("y_true")
        yp = metrics.get("y_pred")

        if yt is not None and yp is not None:
            try:
                yt = np.asarray(yt, dtype=float).ravel()
                yp = np.asarray(yp, dtype=float).ravel()

                if len(yt) == len(yp) and len(yt) > 0:
                    valid = np.isfinite(yt) & np.isfinite(yp)
                    actual = yt[valid]
                    predicted = yp[valid]

                    if len(actual) > 0:
                        pts = pd.DataFrame({
                            "Actual": actual,
                            "Predicted": predicted
                        })

                        fig = px.scatter(
                            pts,
                            x="Actual",
                            y="Predicted",
                            opacity=0.6,
                            labels={
                                "Actual": "Actual price (₹)",
                                "Predicted": "Predicted price (₹)"
                            }
                        )

                        fig.update_traces(marker_color=ACCENT)

                        top = float(max(actual.max(), predicted.max()))

                        fig.add_trace(
                            go.Scatter(
                                x=[0, top],
                                y=[0, top],
                                mode="lines",
                                line=dict(color="#FAB005", dash="dash"),
                                name="Perfect prediction"
                            )
                        )

                        show(
                            style_fig(
                                fig,
                                height=480,
                                title="Actual vs predicted (test sample)"
                            )
                        )

                        # Error by actual-price quartile
                        try:
                            positive = (
                                np.isfinite(actual)
                                & np.isfinite(predicted)
                                & (actual > 0)
                            )

                            actual_pos = actual[positive]
                            predicted_pos = predicted[positive]

                            if len(actual_pos) >= 4:
                                # Wrap in a Series so the .cat accessor works
                                q = pd.Series(
                                    pd.qcut(
                                        actual_pos,
                                        4,
                                        duplicates="drop"
                                    )
                                )

                                ape = (
                                    np.abs(predicted_pos - actual_pos)
                                    / actual_pos
                                )

                                rows = []

                                for cat in q.cat.categories:
                                    sel = np.asarray(q == cat)

                                    if sel.any():
                                        rows.append({
                                            "Price band": (
                                                f"{money_inr(cat.left)} – "
                                                f"{money_inr(cat.right)}"
                                            ),
                                            "Median error %": float(
                                                np.median(ape[sel]) * 100
                                            )
                                        })

                                if rows:
                                    err = pd.DataFrame(rows)

                                    fig = px.bar(
                                        err,
                                        x="Price band",
                                        y="Median error %"
                                    )

                                    fig.update_traces(marker_color=ACCENT_2)

                                    show(
                                        style_fig(
                                            fig,
                                            title="Median absolute % error by price band"
                                        )
                                    )

                                    st.caption(
                                        "This chart shows how prediction errors vary "
                                        "across actual-price bands. Treat it as a guide, "
                                        "not a guarantee for an individual property."
                                    )

                        except (ValueError, TypeError, IndexError, AttributeError):
                            st.info(
                                "Price-band error chart could not be calculated "
                                "for this test sample."
                            )

            except (TypeError, ValueError, OverflowError):
                st.warning(
                    "Actual-versus-predicted chart could not be generated "
                    "from the saved metrics."
                )

    else:
        st.caption(
            "Accuracy cards appear once a trusted metrics file exists."
        )

    # How to generate metrics
    with st.expander("How to generate verified metrics (recommended)"):
        st.markdown(
            "Run this once from the project folder. It recreates the held-out split "
            "and checks the saved evaluation numbers. The settings must match the "
            "original notebook evaluation."
        )

        st.code(
            "python evaluate_model.py --expect-r2 <R²> "
            "--expect-mae <MAE in ₹> --expect-n-test <test rows>",
            language="bash"
        )

        st.markdown(
            "Add `--log-target` if the model was trained on `log1p(Price)`, "
            "and use the same `--test-size` and `--random-state` as your notebook."
        )

    # Feature importance
    section("What drives the predictions")

    perm_imp = (metrics or {}).get("permutation_importance")
    imp = feature_importance(model)

    tabs = st.tabs([
        "Permutation (test data)",
        "Impurity (training)"
    ])

    with tabs[0]:
        if perm_imp:
            p = pd.Series(perm_imp).sort_values().reset_index()
            p.columns = ["Feature", "R² drop when shuffled"]

            fig = px.bar(
                p,
                x="R² drop when shuffled",
                y="Feature",
                orientation="h"
            )

            fig.update_traces(marker_color=ACCENT)

            show(
                style_fig(
                    fig,
                    title="Permutation importance on held-out data"
                )
            )

            st.caption(
                "How much test accuracy falls when a feature is shuffled. "
                "This describes the model's behaviour, not real-world causes."
            )
        else:
            st.info(
                "Run `evaluate_model.py` to add permutation importance to the metrics file."
            )

    with tabs[1]:
        if imp is None:
            st.info(
                "Needs a scikit-learn Pipeline (preprocessor + tree model) "
                "with feature names."
            )
        else:
            imp_df = imp.head(10).sort_values().reset_index()
            imp_df.columns = ["Feature", "Importance"]

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h"
            )

            fig.update_traces(marker_color=ACCENT_2)

            show(
                style_fig(
                    fig,
                    title="Impurity importance (one-hot columns grouped)"
                )
            )

            st.caption(
                "Impurity importance can over-reward high-cardinality features "
                "such as Locality. Prefer the permutation view when available."
            )

    st.caption(
        "Metrics are only meaningful if computed on data the model did not train on. "
        "Load pickled models only from sources you trust."
    )
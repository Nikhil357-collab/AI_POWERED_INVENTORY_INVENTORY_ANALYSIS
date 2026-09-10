from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
import zipfile
import shutil

app = Flask(__name__)
CORS(app)

# ============================================================
# PATHS / DATA SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
ZIP_FILE = BASE_DIR / "data.zip"
EXTRACT_DIR = BASE_DIR / "_render_data"

REQUIRED_FILES = [
    "xgboost_predictions.csv",
    "inventory_risk_scores.csv",
    "reorder_priority_list.csv",
    "markdown_priority_list.csv",
    "risk_summary.csv",
    "sku_master_clean.csv",
    "business_insights.csv",
    "xgmetrics.csv",
    "seasonal_naive_metrics.csv",
    "prioritised_decision_list.csv",
]


def setup_data_directory():
    """Find extracted CSV data locally or inside data.zip on Render."""

    # Local extracted data
    if DATA_DIR.exists():
        found = sum((DATA_DIR / f).is_file() for f in REQUIRED_FILES)
        if found >= 3:
            print(f"Using existing data directory: {DATA_DIR}")
            print(f"Found {found}/{len(REQUIRED_FILES)} required files")
            return DATA_DIR

    # Render/GitHub ZIP fallback
    if ZIP_FILE.exists():
        print(f"Found data.zip: {ZIP_FILE}")

        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)

        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(ZIP_FILE, "r") as z:
                z.extractall(EXTRACT_DIR)
        except Exception as e:
            print(f"ZIP extraction failed: {e}")
            return DATA_DIR

        print(f"Extracted data.zip to: {EXTRACT_DIR}")

        # Find the directory containing the most required files.
        best_dir = None
        best_count = 0

        for root, dirs, files in os.walk(EXTRACT_DIR):
            root_path = Path(root)
            file_set = set(files)
            count = sum(f in file_set for f in REQUIRED_FILES)

            if count > best_count:
                best_count = count
                best_dir = root_path

        if best_dir is not None:
            print(f"Selected data directory: {best_dir}")
            print(f"Found {best_count}/{len(REQUIRED_FILES)} required files")
            return best_dir

        # Last fallback: use extracted root
        return EXTRACT_DIR

    print(f"WARNING: data.zip not found: {ZIP_FILE}")
    return DATA_DIR


DATA_DIR = setup_data_directory()

FILES = {
    "forecast": DATA_DIR / "xgboost_predictions.csv",
    "risk": DATA_DIR / "inventory_risk_scores.csv",
    "reorder": DATA_DIR / "reorder_priority_list.csv",
    "markdown": DATA_DIR / "markdown_priority_list.csv",
    "risk_summary": DATA_DIR / "risk_summary.csv",
    "sku_master": DATA_DIR / "sku_master_clean.csv",
    "insights": DATA_DIR / "business_insights.csv",
    "metrics": DATA_DIR / "xgmetrics.csv",
    "seasonal_metrics": DATA_DIR / "seasonal_naive_metrics.csv",
    "decision": DATA_DIR / "prioritised_decision_list.csv",
}

print("=" * 60)
print("NORTHBAY FORESIGHT DATA CONFIGURATION")
print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("=" * 60)

# ============================================================
# CSV CACHE
# ============================================================

_CACHE = {}
_DASHBOARD_CACHE = None


def load_csv(key):
    """Safely load and cache a CSV."""
    if key not in FILES:
        return pd.DataFrame()

    path = FILES[key]

    if not path.is_file():
        print(f"Missing file [{key}]: {path}")
        return pd.DataFrame()

    if key in _CACHE:
        return _CACHE[key].copy()

    try:
        df = pd.read_csv(path, low_memory=False)
        _CACHE[key] = df.copy()
        print(f"Loaded {key}: {len(df):,} rows x {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Failed loading {key}: {e}")
        return pd.DataFrame()


# ============================================================
# JSON HELPERS
# ============================================================

def clean_records(df):
    """Convert pandas/numpy values into strict JSON-safe values."""
    if df is None or df.empty:
        return []

    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    records = out.to_dict(orient="records")
    safe = []

    for row in records:
        clean = {}

        for key, value in row.items():
            if value is None or value is pd.NA:
                clean[key] = None
                continue

            if isinstance(value, (pd.Timestamp, np.datetime64)):
                try:
                    clean[key] = pd.Timestamp(value).strftime("%Y-%m-%d")
                except Exception:
                    clean[key] = None
                continue

            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
                continue

            if isinstance(value, (np.floating, float)):
                value = float(value)
                clean[key] = value if math.isfinite(value) else None
                continue

            if isinstance(value, np.bool_):
                clean[key] = bool(value)
                continue

            clean[key] = value

        safe.append(clean)

    return safe


def numeric_sum(df, column):
    if df is None or df.empty or column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(df[column], errors="coerce").fillna(0).sum()
    )


def apply_filters(df):
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    sku = request.args.get("sku_id", "").strip()
    store = request.args.get("store_id", "").strip()
    category = request.args.get("category", "").strip()
    risk_type = request.args.get("risk_type", "").strip()

    if sku and "sku_id" in result.columns:
        result = result[result["sku_id"].astype(str) == sku]

    if store and "store_id" in result.columns:
        result = result[result["store_id"].astype(str) == store]

    if category and "category" in result.columns:
        result = result[result["category"].astype(str) == category]

    # Support both names used by earlier risk.py versions.
    if risk_type:
        if "risk_type" in result.columns:
            result = result[result["risk_type"].astype(str).str.upper() == risk_type.upper()]
        elif "risk_level" in result.columns:
            result = result[result["risk_level"].astype(str).str.upper() == risk_type.upper()]

    return result


def get_limit(default=500, maximum=5000):
    try:
        value = int(request.args.get("limit", default))
    except Exception:
        value = default

    return max(1, min(value, maximum))


def limit_df(df, default=500, maximum=5000):
    return df.head(get_limit(default, maximum))


# ============================================================
# HEALTH / FILES
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "service": "NORTHBAY FORESIGHT Flask Scoring API",
        "status": "healthy",
        "data_directory": str(DATA_DIR),
        "files": {key: path.is_file() for key, path in FILES.items()},
        "endpoints": [
            "/", "/health", "/files", "/insights", "/metrics",
            "/forecast", "/risk", "/reorder", "/markdown",
            "/sku/<sku_id>", "/seasonal_metrics",
            "/risk_summary", "/decision", "/dashboard_data"
        ],
    })


@app.get("/files")
def files_endpoint():
    """JSON-safe file diagnostic endpoint."""
    result = {}

    for key, path in FILES.items():
        exists = path.is_file()
        result[key] = {
            "file": path.name,
            "path": str(path),
            "exists": exists,
            "rows": int(len(load_csv(key))) if exists else 0,
        }

    return jsonify(result)


# ============================================================
# INSIGHTS
# ============================================================

@app.get("/insights")
def insights():
    try:
        forecast = load_csv("forecast")
        risk = load_csv("risk")

        result = {
            "forecast_records": int(len(forecast)),
            "risk_records": int(len(risk)),
            "stockout_count": 0,
            "overstock_count": 0,
            "normal_count": 0,
            "stockout_value": 0.0,
            "overstock_value": 0.0,
            "total_value_at_stake": 0.0,
            "risk_counts": {},
        }

        if not risk.empty:
            # Support risk_type and risk_level.
            risk_col = None
            if "risk_type" in risk.columns:
                risk_col = "risk_type"
            elif "risk_level" in risk.columns:
                risk_col = "risk_level"

            if risk_col:
                labels = risk[risk_col].astype(str).str.upper()
                counts = labels.value_counts()

                result["risk_counts"] = {
                    str(k): int(v) for k, v in counts.items()
                }
                result["stockout_count"] = int(counts.get("STOCKOUT", 0))
                result["overstock_count"] = int(counts.get("OVERSTOCK", 0))
                result["normal_count"] = int(counts.get("NORMAL", 0))

            result["stockout_value"] = numeric_sum(risk, "stockout_value")
            result["overstock_value"] = numeric_sum(risk, "overstock_value")

            if "value_at_stake" in risk.columns:
                result["total_value_at_stake"] = numeric_sum(
                    risk, "value_at_stake"
                )
            else:
                result["total_value_at_stake"] = (
                    result["stockout_value"] +
                    result["overstock_value"]
                )

        return jsonify(result)

    except Exception as e:
        print("INSIGHTS ERROR:", repr(e))
        return jsonify({
            "error": "Unable to calculate insights",
            "message": str(e),
            "forecast_records": 0,
            "risk_records": 0,
            "stockout_count": 0,
            "overstock_count": 0,
            "normal_count": 0,
            "stockout_value": 0,
            "overstock_value": 0,
            "total_value_at_stake": 0,
            "risk_counts": {},
        }), 200


# ============================================================
# DATA ENDPOINTS
# ============================================================

@app.get("/metrics")
def metrics():
    df = load_csv("metrics")
    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/seasonal_metrics")
def seasonal_metrics():
    df = load_csv("seasonal_metrics")
    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/risk_summary")
def risk_summary():
    df = load_csv("risk_summary")
    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/decision")
def decision():
    df = load_csv("decision")
    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/forecast")
def forecast():
    try:
        df = load_csv("forecast")

        if df.empty:
            return jsonify({"data": [], "count": 0})

        df = apply_filters(df)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")

        df = limit_df(df, default=500, maximum=5000)

        return jsonify({
            "data": clean_records(df),
            "count": int(len(df))
        })

    except Exception as e:
        print("FORECAST ERROR:", repr(e))
        return jsonify({
            "data": [],
            "count": 0,
            "error": str(e)
        }), 200


@app.get("/risk")
def risk():
    try:
        df = load_csv("risk")

        if df.empty:
            return jsonify({"data": [], "count": 0})

        df = apply_filters(df)

        if "risk_score" in df.columns:
            df["risk_score"] = pd.to_numeric(
                df["risk_score"], errors="coerce"
            )
            df = df.sort_values("risk_score", ascending=False)

        df = limit_df(df, default=1000, maximum=5000)

        return jsonify({
            "data": clean_records(df),
            "count": int(len(df))
        })

    except Exception as e:
        print("RISK ERROR:", repr(e))
        return jsonify({"data": [], "count": 0, "error": str(e)}), 200


def build_risk_fallback(kind):
    risk_df = load_csv("risk")

    if risk_df.empty:
        return pd.DataFrame()

    if "risk_type" in risk_df.columns:
        col = "risk_type"
    elif "risk_level" in risk_df.columns:
        col = "risk_level"
    else:
        return pd.DataFrame()

    wanted = "STOCKOUT" if kind == "reorder" else "OVERSTOCK"

    return risk_df[
        risk_df[col].astype(str).str.upper() == wanted
    ].copy()


@app.get("/reorder")
def reorder():
    try:
        df = load_csv("reorder")

        if df.empty:
            df = build_risk_fallback("reorder")

        df = apply_filters(df)

        if "risk_score" in df.columns:
            df["risk_score"] = pd.to_numeric(
                df["risk_score"], errors="coerce"
            )
            df = df.sort_values("risk_score", ascending=False)

        df = limit_df(df, default=100, maximum=5000)

        return jsonify({
            "data": clean_records(df),
            "count": int(len(df))
        })

    except Exception as e:
        print("REORDER ERROR:", repr(e))
        return jsonify({"data": [], "count": 0, "error": str(e)}), 200


@app.get("/markdown")
def markdown():
    try:
        df = load_csv("markdown")

        if df.empty:
            df = build_risk_fallback("markdown")

        df = apply_filters(df)

        if "risk_score" in df.columns:
            df["risk_score"] = pd.to_numeric(
                df["risk_score"], errors="coerce"
            )
            df = df.sort_values("risk_score", ascending=False)

        df = limit_df(df, default=100, maximum=5000)

        return jsonify({
            "data": clean_records(df),
            "count": int(len(df))
        })

    except Exception as e:
        print("MARKDOWN ERROR:", repr(e))
        return jsonify({"data": [], "count": 0, "error": str(e)}), 200


@app.get("/sku/<sku_id>")
def sku(sku_id):
    try:
        sku_id = str(sku_id).strip()

        master = load_csv("sku_master")
        forecast_df = load_csv("forecast")
        risk_df = load_csv("risk")

        product = pd.DataFrame()
        f = pd.DataFrame()
        r = pd.DataFrame()

        if not master.empty and "sku_id" in master.columns:
            product = master[
                master["sku_id"].astype(str) == sku_id
            ].copy()

        if not forecast_df.empty and "sku_id" in forecast_df.columns:
            f = forecast_df[
                forecast_df["sku_id"].astype(str) == sku_id
            ].copy()

        if not risk_df.empty and "sku_id" in risk_df.columns:
            r = risk_df[
                risk_df["sku_id"].astype(str) == sku_id
            ].copy()

        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"], errors="coerce")
            f = f.sort_values("date")

        f = f.head(500)

        return jsonify({
            "sku_id": sku_id,
            "found": bool(
                not product.empty or not f.empty or not r.empty
            ),
            "product": clean_records(product),
            "forecast": clean_records(f),
            "risk": clean_records(r),
        })

    except Exception as e:
        print("SKU ERROR:", repr(e))
        return jsonify({
            "sku_id": str(sku_id),
            "found": False,
            "product": [],
            "forecast": [],
            "risk": [],
            "error": str(e),
        }), 200


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_exception(e):
    """Always return JSON instead of an HTML 500 page."""
    print("UNHANDLED API ERROR:", repr(e))
    return jsonify({
        "error": "Internal API error",
        "message": str(e),
    }), 500


# ============================================================
# DASHBOARD DATA - FILTERING, RANKING & ACTION ENGINE
# ============================================================

def _first_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _num_series(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def build_dashboard_data():
    """Build one compact, deployment-safe dataset for the browser dashboard."""
    forecast = load_csv("forecast")
    risk = load_csv("risk")
    master = load_csv("sku_master")

    if forecast.empty and risk.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # ---------- Product master ----------
    m = master.copy()
    if not m.empty and "sku_id" in m.columns:
        m["sku_id"] = m["sku_id"].astype(str)
        keep = [c for c in ["sku_id", "sku_name", "category", "subcategory", "brand", "unit_price", "cost_price"] if c in m.columns]
        m = m[keep].drop_duplicates("sku_id")

    # ---------- Forecast aggregation ----------
    f = forecast.copy()
    if not f.empty:
        f["sku_id"] = f["sku_id"].astype(str) if "sku_id" in f.columns else "UNKNOWN"
        if "store_id" not in f.columns:
            f["store_id"] = "ALL"
        f["store_id"] = f["store_id"].astype(str)
        f["prediction"] = _num_series(f, "prediction")
        if "units_sold" in f.columns:
            f["units_sold"] = _num_series(f, "units_sold")
        else:
            f["units_sold"] = 0.0
        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"], errors="coerce")

        group_cols = ["store_id", "sku_id"]
        agg = f.groupby(group_cols, as_index=False).agg(
            forecast_units=("prediction", "sum"),
            actual_units=("units_sold", "sum"),
            avg_daily_forecast=("prediction", "mean"),
            forecast_days=("prediction", "count"),
        )
    else:
        agg = pd.DataFrame(columns=["store_id", "sku_id", "forecast_units", "actual_units", "avg_daily_forecast", "forecast_days"])

    # ---------- Risk ----------
    r = risk.copy()
    if not r.empty:
        if "sku_id" in r.columns:
            r["sku_id"] = r["sku_id"].astype(str)
        else:
            r["sku_id"] = "UNKNOWN"
        if "store_id" not in r.columns:
            r["store_id"] = "ALL"
        r["store_id"] = r["store_id"].astype(str)
        risk_type_col = _first_col(r, ["risk_type", "risk_level"])
        if risk_type_col:
            r["risk_type_clean"] = r[risk_type_col].astype(str).str.upper()
        else:
            r["risk_type_clean"] = "NORMAL"

        for c in ["risk_score", "stockout_value", "overstock_value", "reorder_qty", "shortage_units", "excess_units", "stock_on_hand", "unit_price", "cost_price"]:
            if c in r.columns:
                r[c] = _num_series(r, c)

        # Prefer the strongest risk row per store/SKU.
        if "risk_score" in r.columns:
            r = r.sort_values("risk_score", ascending=False)
        r = r.drop_duplicates(["store_id", "sku_id"], keep="first")
        risk_keep = [c for c in [
            "store_id", "sku_id", "risk_type_clean", "risk_score", "priority",
            "recommended_action", "stockout_value", "overstock_value", "reorder_qty",
            "shortage_units", "excess_units", "stock_on_hand", "reorder_point", "safety_stock"
        ] if c in r.columns]
        r = r[risk_keep]
    else:
        r = pd.DataFrame(columns=["store_id", "sku_id", "risk_type_clean", "risk_score"])

    # ---------- Combine ----------
    if agg.empty:
        base = r.copy()
    elif r.empty:
        base = agg.copy()
    else:
        base = agg.merge(r, on=["store_id", "sku_id"], how="outer")

    if "sku_id" not in base.columns:
        base["sku_id"] = "UNKNOWN"
    if "store_id" not in base.columns:
        base["store_id"] = "ALL"
    base["sku_id"] = base["sku_id"].astype(str)
    base["store_id"] = base["store_id"].astype(str)

    if not m.empty:
        base = base.merge(m, on="sku_id", how="left", suffixes=("", "_master"))

    for c in ["forecast_units", "actual_units", "avg_daily_forecast", "risk_score", "stockout_value", "overstock_value", "reorder_qty", "shortage_units", "excess_units", "stock_on_hand", "unit_price", "cost_price"]:
        if c in base.columns:
            base[c] = _num_series(base, c)

    for c in ["sku_name", "category", "subcategory", "brand"]:
        if c not in base.columns:
            base[c] = "Unknown"
        base[c] = base[c].fillna("Unknown").astype(str)

    base["unit_price"] = _num_series(base, "unit_price")
    base["cost_price"] = _num_series(base, "cost_price")
    base["forecast_revenue"] = base.get("forecast_units", 0) * base["unit_price"]
    base["forecast_profit"] = base.get("forecast_units", 0) * (base["unit_price"] - base["cost_price"])
    base["actual_revenue"] = base.get("actual_units", 0) * base["unit_price"]
    base["profit_margin_pct"] = np.where(
        base["forecast_revenue"] > 0,
        (base["forecast_profit"] / base["forecast_revenue"]) * 100,
        0,
    )

    # ---------- Transparent business action (vectorized for Render speed) ----------
    existing_action = base.get("recommended_action", pd.Series("", index=base.index)).fillna("").astype(str).str.upper().str.strip()
    risk_type = base.get("risk_type_clean", pd.Series("NORMAL", index=base.index)).fillna("NORMAL").astype(str).str.upper().str.strip()
    score = _num_series(base, "risk_score")
    reorder_qty = _num_series(base, "reorder_qty")
    excess = _num_series(base, "excess_units")

    valid_existing = existing_action.ne("") & ~existing_action.isin(["NAN", "NONE", "NO ACTION"])
    urgent_reorder = (risk_type.eq("STOCKOUT") | reorder_qty.gt(0)) & score.ge(75)
    normal_reorder = (risk_type.eq("STOCKOUT") | reorder_qty.gt(0))
    urgent_markdown = (risk_type.eq("OVERSTOCK") | excess.gt(0)) & score.ge(75)
    normal_markdown = (risk_type.eq("OVERSTOCK") | excess.gt(0))

    inferred = np.select(
        [urgent_reorder, normal_reorder, urgent_markdown, normal_markdown, score.ge(50)],
        ["URGENT REORDER", "REORDER", "URGENT MARKDOWN", "MARKDOWN / SELL NOW", "WATCH CLOSELY"],
        default="NO ACTION",
    )
    base["action"] = np.where(valid_existing, existing_action, inferred)
    base["red_flag"] = np.where(
        risk_type.isin(["STOCKOUT", "OVERSTOCK"]) | score.ge(75),
        "RED FLAG",
        np.where(score.ge(50), "WATCH", "OK"),
    )
    base["value_at_stake"] = _num_series(base, "stockout_value") + _num_series(base, "overstock_value")

    # Ranking at product level (across stores).
    product = base.groupby(["sku_id", "sku_name", "category", "subcategory", "brand"], as_index=False).agg(
        forecast_units=("forecast_units", "sum"),
        actual_units=("actual_units", "sum"),
        forecast_revenue=("forecast_revenue", "sum"),
        forecast_profit=("forecast_profit", "sum"),
        value_at_stake=("value_at_stake", "sum"),
        stockout_value=("stockout_value", "sum"),
        overstock_value=("overstock_value", "sum"),
        max_risk_score=("risk_score", "max"),
    )
    product["profit_margin_pct"] = np.where(product["forecast_revenue"] > 0, product["forecast_profit"] / product["forecast_revenue"] * 100, 0)
    product["red_flag"] = np.where(product["max_risk_score"] >= 75, "RED FLAG", np.where(product["max_risk_score"] >= 50, "WATCH", "OK"))

    meta = {
        "categories": sorted([x for x in base["category"].dropna().unique().tolist() if x not in ["", "Unknown"]]),
        "stores": sorted([x for x in base["store_id"].dropna().unique().tolist() if x not in ["", "Unknown"]]),
        "risk_types": sorted([x for x in base["risk_type_clean"].dropna().unique().tolist()]),
        "actions": sorted([x for x in base["action"].dropna().unique().tolist()]),
        "skus": sorted(base["sku_id"].dropna().unique().tolist())[:10000],
    }
    return base, product, meta


@app.get("/dashboard_data")
def dashboard_data():
    """Single cached endpoint used by the Flask dashboard."""
    global _DASHBOARD_CACHE
    try:
        if _DASHBOARD_CACHE is None:
            _DASHBOARD_CACHE = build_dashboard_data()
        base, product, meta = _DASHBOARD_CACHE
        if base.empty:
            return jsonify({"data": [], "products": [], "meta": meta, "count": 0})

        # Filters
        category = request.args.get("category", "").strip()
        sku = request.args.get("sku_id", "").strip()
        store = request.args.get("store_id", "").strip()
        risk_type = request.args.get("risk_type", "").strip().upper()
        action = request.args.get("action", "").strip().upper()
        red_flag = request.args.get("red_flag", "").strip().upper()
        ranking = request.args.get("ranking", "revenue").strip().lower()
        sort_dir = request.args.get("sort", "desc").strip().lower()
        limit = get_limit(default=250, maximum=3000)

        if category:
            base = base[base["category"].str.casefold() == category.casefold()]
        if sku:
            base = base[base["sku_id"].astype(str) == sku]
        if store:
            base = base[base["store_id"].astype(str) == store]
        if risk_type:
            base = base[base["risk_type_clean"].astype(str).str.upper() == risk_type]
        if action:
            base = base[base["action"].astype(str).str.upper() == action]
        if red_flag:
            base = base[base["red_flag"].astype(str).str.upper() == red_flag]

        # Product table follows the same filters where relevant.
        if category:
            product = product[product["category"].str.casefold() == category.casefold()]
        if sku:
            product = product[product["sku_id"].astype(str) == sku]

        rank_col = {
            "revenue": "forecast_revenue",
            "profit": "forecast_profit",
            "low_profit": "forecast_profit",
            "risk": "max_risk_score",
            "value": "value_at_stake",
        }.get(ranking, "forecast_revenue")
        ascending = sort_dir == "asc"
        product = product.sort_values(rank_col, ascending=ascending).head(50)

        base = base.sort_values(["red_flag", "risk_score", "value_at_stake"], ascending=[True, False, False]).head(limit)

        # KPI calculations on the filtered base.
        kpis = {
            "products": int(base["sku_id"].nunique()),
            "store_sku": int(len(base)),
            "forecast_units": float(base["forecast_units"].sum()),
            "revenue": float(base["forecast_revenue"].sum()),
            "profit": float(base["forecast_profit"].sum()),
            "value_at_stake": float(base["value_at_stake"].sum()),
            "stockout_value": float(base["stockout_value"].sum()),
            "overstock_value": float(base["overstock_value"].sum()),
            "red_flags": int((base["red_flag"] == "RED FLAG").sum()),
            "reorder": int(base["action"].str.contains("REORDER", na=False).sum()),
            "markdown": int(base["action"].str.contains("MARKDOWN|SELL NOW", regex=True, na=False).sum()),
        }

        risk_counts = base["risk_type_clean"].value_counts().to_dict()
        action_counts = base["action"].value_counts().to_dict()

        # Keep response compact and JSON-safe.
        columns = [c for c in [
            "store_id", "sku_id", "sku_name", "category", "subcategory", "brand",
            "forecast_units", "actual_units", "forecast_revenue", "forecast_profit",
            "profit_margin_pct", "stock_on_hand", "risk_type_clean", "risk_score",
            "priority", "reorder_qty", "shortage_units", "excess_units", "stockout_value",
            "overstock_value", "value_at_stake", "action", "red_flag"
        ] if c in base.columns]
        pcols = [c for c in [
            "sku_id", "sku_name", "category", "subcategory", "brand", "forecast_units",
            "actual_units", "forecast_revenue", "forecast_profit", "profit_margin_pct",
            "value_at_stake", "stockout_value", "overstock_value", "max_risk_score", "red_flag"
        ] if c in product.columns]

        return jsonify({
            "data": clean_records(base[columns]),
            "products": clean_records(product[pcols]),
            "meta": meta,
            "kpis": kpis,
            "risk_counts": {str(k): int(v) for k, v in risk_counts.items()},
            "action_counts": {str(k): int(v) for k, v in action_counts.items()},
            "count": int(len(base)),
        })
    except Exception as e:
        print("DASHBOARD DATA ERROR:", repr(e))
        return jsonify({"error": "Dashboard data error", "message": str(e), "data": [], "products": [], "meta": {}}), 200


# ============================================================
# FLASK DASHBOARD
# ============================================================

DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NORTHBAY FORESIGHT | AI Retail Control Tower</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
  --bg:rgb(8,12,24); --panel:rgb(17,24,39); --panel2:rgb(23,32,52);
  --text:rgb(241,245,249); --muted:rgb(148,163,184); --line:rgb(51,65,85);
  --cyan:rgb(34,211,238); --green:rgb(52,211,153); --amber:rgb(251,191,36);
  --red:rgb(248,113,113); --blue:rgb(96,165,250); --purple:rgb(167,139,250);
}
*{box-sizing:border-box} body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:linear-gradient(135deg,rgb(7,12,25),rgb(15,23,42),rgb(12,20,38));color:var(--text)}
.wrap{max-width:1500px;margin:auto;padding:24px}.hero{padding:26px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(135deg,rgb(17,24,39),rgb(22,32,55));box-shadow:0 18px 50px rgba(0,0,0,.28)}
h1{margin:0;font-size:30px}.sub{color:var(--muted);margin-top:7px}.status{display:inline-flex;margin-top:15px;padding:7px 12px;border-radius:999px;background:rgb(22,101,52);color:white;font-size:13px}.status.warn{background:rgb(146,64,14)}
.filters{margin-top:18px;display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:10px}.filters select,.filters input{width:100%;padding:11px;border-radius:11px;border:1px solid var(--line);background:rgb(15,23,42);color:var(--text);outline:none}.filters button{padding:11px;border:0;border-radius:11px;background:linear-gradient(135deg,rgb(6,182,212),rgb(59,130,246));color:white;font-weight:700;cursor:pointer}.btn2{background:rgb(51,65,85)!important}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}.kpi{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:17px;padding:16px}.kpi small{color:var(--muted)}.kpi b{display:block;font-size:22px;margin-top:6px}.danger b{color:var(--red)}.good b{color:var(--green)}.money b{color:var(--cyan)}
.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:14px}.card{background:rgba(17,24,39,.88);border:1px solid var(--line);border-radius:18px;padding:16px;margin-bottom:14px}.card h2{font-size:17px;margin:0 0 12px}.chart{height:330px}.wide{grid-column:1/-1}.tablewrap{overflow:auto;max-height:430px;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px;border-bottom:1px solid rgb(30,41,59);white-space:nowrap;text-align:left}th{position:sticky;top:0;background:rgb(15,23,42);z-index:1;color:var(--muted)}tr:hover{background:rgb(30,41,59)}.red{color:var(--red);font-weight:800}.green{color:var(--green);font-weight:700}.amber{color:var(--amber);font-weight:700}.pill{padding:4px 8px;border-radius:999px;background:rgb(51,65,85)}.note{color:var(--muted);font-size:12px;line-height:1.5}.error{color:var(--red)}
@media(max-width:1100px){.filters{grid-template-columns:repeat(3,1fr)}.kpis{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:650px){.wrap{padding:12px}.filters{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">
 <section class="hero">
  <h1>NorthBay Foresight — AI Retail Control Tower</h1>
  <div class="sub">Demand forecast • revenue & profit ranking • inventory risk • action recommendations</div>
  <span id="status" class="status">Connecting…</span>
  <div class="filters">
   <select id="category"><option value="">All Categories</option></select>
   <select id="sku"><option value="">All Product / SKU</option></select>
   <select id="store"><option value="">All Stores</option></select>
   <select id="risk"><option value="">All Risk Types</option></select>
   <select id="action"><option value="">All Actions</option></select>
   <select id="flag"><option value="">All Flags</option><option>RED FLAG</option><option>WATCH</option><option>OK</option></select>
   <select id="ranking"><option value="revenue">Highest Revenue</option><option value="profit">Highest Profit</option><option value="low_profit">Lowest Profit</option><option value="risk">Highest Risk</option><option value="value">Highest Value at Stake</option></select>
   <input id="search" placeholder="Search product name / brand">
   <button onclick="loadDashboard()">Apply Filters</button>
   <button class="btn2" onclick="resetFilters()">Reset</button>
  </div>
 </section>

 <section class="kpis">
  <div class="kpi"><small>Products</small><b id="kProducts">—</b></div>
  <div class="kpi"><small>Forecast Units</small><b id="kUnits">—</b></div>
  <div class="kpi money"><small>Forecast Revenue</small><b id="kRevenue">—</b></div>
  <div class="kpi good"><small>Forecast Profit</small><b id="kProfit">—</b></div>
  <div class="kpi danger"><small>Value at Stake</small><b id="kStake">—</b></div>
  <div class="kpi danger"><small>Red Flags</small><b id="kFlags">—</b></div>
 </section>

 <div class="grid">
  <section class="card"><h2>Revenue vs Profit — Top Products</h2><div id="productChart" class="chart"></div></section>
  <section class="card"><h2>Risk Distribution</h2><div id="riskChart" class="chart"></div></section>
  <section class="card"><h2>Business Actions</h2><div id="actionChart" class="chart"></div></section>
  <section class="card"><h2>Value at Stake</h2><div id="valueChart" class="chart"></div></section>

  <section class="card wide"><h2>🚨 Priority Decision Grid — what to do now</h2><div class="note">REORDER = stock is at risk; MARKDOWN / SELL NOW = excess inventory needs faster sales; WATCH = monitor closely; NO ACTION = currently stable. Red Flag means immediate attention.</div><div id="decisionTable" class="tablewrap"></div></section>
  <section class="card wide"><h2>🏆 Product Ranking</h2><div id="productTable" class="tablewrap"></div></section>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const money=x=>'₹'+Number(x||0).toLocaleString('en-IN',{maximumFractionDigits:0});
const num=x=>Number(x||0).toLocaleString('en-IN',{maximumFractionDigits:1});
let allProducts=[];
function fill(id, values, label){const el=$(id); const first=el.options[0]; el.innerHTML=''; el.appendChild(first); values.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)});}
function setK(id,v){$(id).textContent=v}
function tableHtml(rows, product=false){
 if(!rows||!rows.length)return '<div style="padding:20px;color:rgb(148,163,184)">No records match the selected filters.</div>';
 const cols=product?['sku_id','sku_name','category','forecast_units','forecast_revenue','forecast_profit','profit_margin_pct','value_at_stake','max_risk_score','red_flag']:['store_id','sku_id','sku_name','category','forecast_units','forecast_revenue','forecast_profit','risk_type_clean','risk_score','stock_on_hand','reorder_qty','action','value_at_stake','red_flag'];
 let h='<table><thead><tr>'+cols.map(c=>'<th>'+c.replaceAll('_',' ').toUpperCase()+'</th>').join('')+'</tr></thead><tbody>';
 rows.forEach(r=>{h+='<tr>'+cols.map(c=>{let v=r[c];if(['forecast_revenue','forecast_profit','value_at_stake'].includes(c))v=money(v);else if(['forecast_units','stock_on_hand','reorder_qty'].includes(c))v=num(v);else if(['risk_score','max_risk_score','profit_margin_pct'].includes(c))v=num(v)+'%';let cls=(String(v).includes('RED FLAG')||String(r.action||'').includes('URGENT'))?'red':(String(r.action||'').includes('MARKDOWN')?'amber':(String(r.action||'').includes('REORDER')?'red':''));return '<td class="'+cls+'">'+(v??'—')+'</td>'}).join('')+'</tr>'});
 return h+'</tbody></table>';
}
function plot(id,data,layout){if(typeof Plotly==='undefined'){return} Plotly.newPlot(id,data,layout,{responsive:true,displayModeBar:false});}
async function fetchJson(url,attempt=0){
 const res=await fetch(url,{cache:'no-store'});
 const text=await res.text();
 if(!res.ok){
   if(attempt<2){await new Promise(r=>setTimeout(r,1200*(attempt+1)));return fetchJson(url,attempt+1);}
   throw new Error('HTTP '+res.status+(text?' — '+text.slice(0,180):' — empty response'));
 }
 if(!text.trim()){
   if(attempt<2){await new Promise(r=>setTimeout(r,1200*(attempt+1)));return fetchJson(url,attempt+1);}
   throw new Error('Server returned an empty response');
 }
 try{return JSON.parse(text)}catch(e){throw new Error('Server returned invalid JSON: '+text.slice(0,180))}
}
async function loadDashboard(){
 const q=new URLSearchParams(); ['category','sku','store','risk','action','flag','ranking'].forEach(id=>{let v=$(id).value;if(v)q.set(id==='risk'?'risk_type':id==='flag'?'red_flag':id==='sku'?'sku_id':id,v)});
 q.set('limit','1000');
 $('status').textContent='Loading dashboard data…'; $('status').className='status';
 try{
  const json=await fetchJson('/dashboard_data?'+q.toString());
  if(json.error)throw new Error(json.message||json.error);
  const k=json.kpis||{}; setK('kProducts',num(k.products));setK('kUnits',num(k.forecast_units));setK('kRevenue',money(k.revenue));setK('kProfit',money(k.profit));setK('kStake',money(k.value_at_stake));setK('kFlags',num(k.red_flags));
  $('status').textContent='✓ Dashboard connected • '+num(json.count)+' filtered store-SKU records'; $('status').className='status';
  if(json.meta){
   if(!$('category').dataset.loaded){fill('category',json.meta.categories||[]);fill('store',json.meta.stores||[]);fill('risk',json.meta.risk_types||[]);fill('action',json.meta.actions||[]);fill('sku',json.meta.skus||[]);$('category').dataset.loaded='1';}
  }
  const products=json.products||[];
  plot('productChart',[{x:products.slice(0,12).map(x=>x.sku_id),y:products.slice(0,12).map(x=>x.forecast_revenue),type:'bar',name:'Revenue'},{x:products.slice(0,12).map(x=>x.sku_id),y:products.slice(0,12).map(x=>x.forecast_profit),type:'bar',name:'Profit'}],{barmode:'group',margin:{t:10,l:65,r:10,b:70},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},yaxis:{tickprefix:'₹'}});
  const rc=json.risk_counts||{};plot('riskChart',[{labels:Object.keys(rc),values:Object.values(rc),type:'pie',hole:.55}],{margin:{t:10,l:10,r:10,b:10},paper_bgcolor:'transparent',font:{color:'rgb(226,232,240)'}});
  const ac=json.action_counts||{};plot('actionChart',[{x:Object.keys(ac),y:Object.values(ac),type:'bar'}],{margin:{t:10,l:10,r:10,b:80},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},xaxis:{tickangle:-25}});
  plot('valueChart',[{x:products.slice(0,10).map(x=>x.sku_id),y:products.slice(0,10).map(x=>x.value_at_stake),type:'bar'}],{margin:{t:10,l:65,r:10,b:70},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},yaxis:{tickprefix:'₹'}});
  let rows=json.data||[]; const term=$('search').value.trim().toLowerCase(); if(term)rows=rows.filter(r=>(String(r.sku_name)+' '+String(r.brand)).toLowerCase().includes(term));
  rows.sort((a,b)=>{let af=String(a.red_flag),bf=String(b.red_flag);return af.localeCompare(bf)||Number(b.risk_score||0)-Number(a.risk_score||0)});
  $('decisionTable').innerHTML=tableHtml(rows,false);$('productTable').innerHTML=tableHtml(products,true);
 }catch(e){$('status').textContent='⚠ '+e.message;$('status').className='status warn';$('decisionTable').innerHTML='<div class="error" style="padding:18px">Dashboard data could not be loaded: '+e.message+'</div>';}
}
function resetFilters(){['category','sku','store','risk','action','flag'].forEach(id=>$(id).selectedIndex=0);$('ranking').selectedIndex=0;$('search').value='';loadDashboard()}
loadDashboard();
</script>
</body></html>
"""

@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

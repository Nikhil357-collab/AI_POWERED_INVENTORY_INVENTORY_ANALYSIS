
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import os
import zipfile
import shutil
import json
import time

# ============================================================
# NORTHBAY FORESIGHT - DEPLOYMENT-SAFE FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
ZIP_FILE = BASE_DIR / "data.zip"
EXTRACT_DIR = BASE_DIR / "_render_data"

REQUIRED_FILES = [
    "xgboost_predictions.csv",
    "inventory_risk_scores.csv",
    "sku_master_clean.csv",
]

FILES = {}

_CACHE = {}
_DASHBOARD_CACHE = None
_DASHBOARD_BUILD_ERROR = None
_DASHBOARD_BUILD_SECONDS = 0.0


def setup_data_directory():
    """Locate CSVs either directly in the repo or inside data.zip."""
    if DATA_DIR.exists():
        found = sum((DATA_DIR / f).is_file() for f in REQUIRED_FILES)
        if found >= 3:
            print(f"DATA: using {DATA_DIR} ({found}/{len(REQUIRED_FILES)} core files)")
            return DATA_DIR

    if ZIP_FILE.exists():
        print(f"DATA: extracting {ZIP_FILE}")
        try:
            if EXTRACT_DIR.exists():
                shutil.rmtree(EXTRACT_DIR)
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_FILE, "r") as z:
                z.extractall(EXTRACT_DIR)
        except Exception as e:
            print("DATA ZIP ERROR:", repr(e))
            return DATA_DIR

        best_dir = EXTRACT_DIR
        best_count = -1
        for root, _, files in os.walk(EXTRACT_DIR):
            file_set = set(files)
            count = sum(x in file_set for x in REQUIRED_FILES)
            if count > best_count:
                best_count = count
                best_dir = Path(root)

        print(f"DATA: selected {best_dir} ({max(best_count, 0)}/{len(REQUIRED_FILES)} core files)")
        return best_dir

    print("DATA WARNING: data.zip not found")
    return DATA_DIR


DATA_DIR = setup_data_directory()

# Map every optional file too. Missing optional files do not break the app.
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

print("=" * 65)
print("NORTHBAY FORESIGHT")
print("BASE_DIR :", BASE_DIR)
print("DATA_DIR :", DATA_DIR)
for k, p in FILES.items():
    print(f"{k:16s}: {'OK' if p.is_file() else 'MISSING'} -> {p.name}")
print("=" * 65)


# ============================================================
# SAFE CSV / JSON HELPERS
# ============================================================

def load_csv(key):
    if key not in FILES:
        return pd.DataFrame()
    path = FILES[key]
    if not path.is_file():
        return pd.DataFrame()
    if key in _CACHE:
        return _CACHE[key].copy()
    try:
        df = pd.read_csv(path, low_memory=False)
        _CACHE[key] = df.copy()
        print(f"CSV: loaded {key}: {len(df):,} rows x {len(df.columns)} cols")
        return df
    except Exception as e:
        print(f"CSV ERROR [{key}]:", repr(e))
        return pd.DataFrame()


def num_col(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_value(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except Exception:
            return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def records(df):
    if df is None or df.empty:
        return []
    out = []
    for row in df.to_dict(orient="records"):
        out.append({str(k): clean_value(v) for k, v in row.items()})
    return out


def limit_value(default=100, maximum=1000):
    try:
        n = int(request.args.get("limit", default))
    except Exception:
        n = default
    return max(1, min(n, maximum))


def safe_json(payload, status=200):
    # Flask's jsonify handles normal dict/list values. This function also
    # protects against accidental numpy values in endpoint responses.
    try:
        return jsonify(payload), status
    except Exception as e:
        return jsonify({"error": "JSON serialization error", "message": str(e)}), 500


# ============================================================
# SIMPLE API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return safe_json({
        "status": "ok",
        "service": "NorthBay Foresight Flask",
        "data_dir": str(DATA_DIR),
        "dashboard_cache": _DASHBOARD_CACHE is not None,
        "dashboard_build_error": _DASHBOARD_BUILD_ERROR,
        "dashboard_build_seconds": round(_DASHBOARD_BUILD_SECONDS, 3),
        "files": {
            k: bool(p.is_file())
            for k, p in FILES.items()
        },
    })


@app.get("/files")
def files_endpoint():
    result = {}
    for key, path in FILES.items():
        exists = path.is_file()
        result[key] = {
            "file": path.name,
            "path": str(path),
            "exists": exists,
            "rows": int(len(load_csv(key))) if exists else 0,
        }
    return safe_json(result)


@app.get("/forecast")
def forecast_endpoint():
    df = load_csv("forecast")
    if df.empty:
        return safe_json({"data": [], "count": 0, "message": "Forecast file unavailable"}, 200)
    if "sku_id" in df.columns:
        sku = request.args.get("sku_id", "").strip()
        if sku:
            df = df[df["sku_id"].astype(str) == sku]
    if "store_id" in df.columns:
        store = request.args.get("store_id", "").strip()
        if store:
            df = df[df["store_id"].astype(str) == store]
    return safe_json({"data": records(df.head(limit_value(100, 1000))), "count": int(len(df))})


@app.get("/risk")
def risk_endpoint():
    df = load_csv("risk")
    if df.empty:
        return safe_json({"data": [], "count": 0, "message": "Risk file unavailable"}, 200)
    sku = request.args.get("sku_id", "").strip()
    store = request.args.get("store_id", "").strip()
    if sku and "sku_id" in df.columns:
        df = df[df["sku_id"].astype(str) == sku]
    if store and "store_id" in df.columns:
        df = df[df["store_id"].astype(str) == store]
    return safe_json({"data": records(df.head(limit_value(100, 1000))), "count": int(len(df))})


def generic_csv_endpoint(key):
    df = load_csv(key)
    return safe_json({"data": records(df.head(limit_value(100, 1000))), "count": int(len(df))})


@app.get("/reorder")
def reorder_endpoint():
    return generic_csv_endpoint("reorder")


@app.get("/markdown")
def markdown_endpoint():
    return generic_csv_endpoint("markdown")


@app.get("/metrics")
def metrics_endpoint():
    return generic_csv_endpoint("metrics")


@app.get("/seasonal_metrics")
def seasonal_metrics_endpoint():
    return generic_csv_endpoint("seasonal_metrics")


@app.get("/risk_summary")
def risk_summary_endpoint():
    return generic_csv_endpoint("risk_summary")


@app.get("/decision")
def decision_endpoint():
    return generic_csv_endpoint("decision")


@app.get("/insights")
def insights_endpoint():
    df = load_csv("insights")
    if df.empty:
        return safe_json({"data": [], "count": 0, "message": "Business insights file unavailable"})
    return safe_json({"data": records(df.head(100)), "count": int(len(df))})


@app.get("/sku/<sku_id>")
def sku_endpoint(sku_id):
    sku_id = str(sku_id)
    master = load_csv("sku_master")
    forecast = load_csv("forecast")
    risk = load_csv("risk")

    product = master[master["sku_id"].astype(str) == sku_id] if not master.empty and "sku_id" in master.columns else pd.DataFrame()
    f = forecast[forecast["sku_id"].astype(str) == sku_id].copy() if not forecast.empty and "sku_id" in forecast.columns else pd.DataFrame()
    r = risk[risk["sku_id"].astype(str) == sku_id].copy() if not risk.empty and "sku_id" in risk.columns else pd.DataFrame()

    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.sort_values("date")

    return safe_json({
        "sku_id": sku_id,
        "found": bool(not product.empty or not f.empty or not r.empty),
        "product": records(product.head(1)),
        "forecast": records(f.head(500)),
        "risk": records(r.head(100)),
    })


# ============================================================
# DASHBOARD DATA ENGINE
# ============================================================

def build_dashboard_data():
    """
    Build ONE compact dataset from the two operational files and product master.
    This is deliberately done once, not once per browser request.
    """
    started = time.time()

    forecast = load_csv("forecast")
    risk = load_csv("risk")
    master = load_csv("sku_master")

    if forecast.empty and risk.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # ---------- Forecast: store + SKU aggregation ----------
    if forecast.empty:
        fagg = pd.DataFrame(columns=[
            "store_id", "sku_id", "forecast_units", "actual_units",
            "avg_daily_forecast", "forecast_days"
        ])
    else:
        f = forecast.copy()
        f["sku_id"] = f["sku_id"].astype(str) if "sku_id" in f.columns else "UNKNOWN"
        f["store_id"] = f["store_id"].astype(str) if "store_id" in f.columns else "ALL"
        f["prediction"] = num_col(f, "prediction")
        f["units_sold"] = num_col(f, "units_sold")
        fagg = f.groupby(["store_id", "sku_id"], as_index=False).agg(
            forecast_units=("prediction", "sum"),
            actual_units=("units_sold", "sum"),
            avg_daily_forecast=("prediction", "mean"),
            forecast_days=("prediction", "count"),
        )

    # ---------- Risk: one strongest row per store + SKU ----------
    if risk.empty:
        r = pd.DataFrame(columns=["store_id", "sku_id", "risk_type_clean", "risk_score"])
    else:
        r = risk.copy()
        r["sku_id"] = r["sku_id"].astype(str) if "sku_id" in r.columns else "UNKNOWN"
        r["store_id"] = r["store_id"].astype(str) if "store_id" in r.columns else "ALL"

        rt = first_col(r, ["risk_type", "risk_level"])
        r["risk_type_clean"] = r[rt].astype(str).str.upper() if rt else "NORMAL"

        for c in [
            "risk_score", "stockout_value", "overstock_value", "reorder_qty",
            "shortage_units", "excess_units", "stock_on_hand", "reorder_point",
            "safety_stock"
        ]:
            if c in r.columns:
                r[c] = num_col(r, c)

        if "risk_score" in r.columns:
            r = r.sort_values("risk_score", ascending=False)

        r = r.drop_duplicates(["store_id", "sku_id"], keep="first")

    # ---------- Merge ----------
    if fagg.empty:
        base = r.copy()
    elif r.empty:
        base = fagg.copy()
    else:
        base = fagg.merge(r, on=["store_id", "sku_id"], how="outer")

    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    base["sku_id"] = base["sku_id"].astype(str)
    base["store_id"] = base["store_id"].astype(str)

    # ---------- Product master ----------
    if not master.empty and "sku_id" in master.columns:
        m = master.copy()
        m["sku_id"] = m["sku_id"].astype(str)
        keep = [
            c for c in [
                "sku_id", "sku_name", "category", "subcategory",
                "brand", "unit_price", "cost_price"
            ] if c in m.columns
        ]
        m = m[keep].drop_duplicates("sku_id")
        base = base.merge(m, on="sku_id", how="left")

    for c in ["sku_name", "category", "subcategory", "brand"]:
        if c not in base.columns:
            base[c] = "Unknown"
        base[c] = base[c].fillna("Unknown").astype(str)

    for c in [
        "forecast_units", "actual_units", "avg_daily_forecast", "risk_score",
        "stockout_value", "overstock_value", "reorder_qty",
        "shortage_units", "excess_units", "stock_on_hand",
        "reorder_point", "safety_stock", "unit_price", "cost_price"
    ]:
        if c not in base.columns:
            base[c] = 0.0
        base[c] = num_col(base, c)

    # ---------- Financial metrics ----------
    base["forecast_revenue"] = base["forecast_units"] * base["unit_price"]
    base["forecast_profit"] = base["forecast_units"] * (
        base["unit_price"] - base["cost_price"]
    )
    base["profit_margin_pct"] = np.where(
        base["forecast_revenue"] > 0,
        base["forecast_profit"] / base["forecast_revenue"] * 100,
        0,
    )
    #base["value_at_stake"] = base["stockout_value"] + base["overstock_value"]
    def calculate_financial_metrics(df):
    df = df.copy()

    # Convert financial columns safely
    for col in [
        "stockout_value_at_stake",
        "overstock_value_at_stake",
        "unit_price",
        "cost_price",
        "shortage_units",
        "excess_inventory_units"
    ]:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    # Only VALID inventory decisions contribute to business exposure
    valid_risk = df["risk_level"].isin([
        "STOCKOUT",
        "OVERSTOCK"
    ])

    # Stockout exposure
    df["stockout_value_at_stake"] = (
        df["stockout_value_at_stake"]
        .where(df["risk_level"] == "STOCKOUT", 0)
    )

    # Overstock exposure
    df["overstock_value_at_stake"] = (
        df["overstock_value_at_stake"]
        .where(df["risk_level"] == "OVERSTOCK", 0)
    )

    # Total value at stake
    df["value_at_stake"] = (
        df["stockout_value_at_stake"]
        + df["overstock_value_at_stake"]
    )

    # DATA_ISSUE must never contribute financial exposure
    df.loc[
        ~valid_risk,
        "value_at_stake"
    ] = 0

    return df
    # ---------- Transparent decision engine ----------
    risk_type = base["risk_type_clean"].fillna("NORMAL").astype(str).str.upper()
    score = base["risk_score"].clip(lower=0, upper=100)
    reorder_qty = base["reorder_qty"].clip(lower=0)
    excess = base["excess_units"].clip(lower=0)

    # Use existing recommendation only if it contains useful information.
    existing = (
        base["recommended_action"]
        if "recommended_action" in base.columns
        else pd.Series("", index=base.index)
    )
    existing = existing.fillna("").astype(str).str.upper().str.strip()
    valid_existing = existing.ne("") & ~existing.isin(["NAN", "NONE", "NO ACTION"])

    inferred = np.select(
        [
            ((risk_type == "STOCKOUT") | (reorder_qty > 0)) & (score >= 75),
            ((risk_type == "STOCKOUT") | (reorder_qty > 0)),
            ((risk_type == "OVERSTOCK") | (excess > 0)) & (score >= 75),
            ((risk_type == "OVERSTOCK") | (excess > 0)),
            score >= 50,
        ],
        [
            "URGENT REORDER",
            "REORDER",
            "URGENT MARKDOWN / SELL NOW",
            "MARKDOWN / SELL NOW",
            "WATCH CLOSELY",
        ],
        default="NO ACTION",
    )
    base["action"] = np.where(valid_existing, existing, inferred)

    # Red flag = genuinely urgent, not every normal stockout/overstock row.
    base["red_flag"] = np.select(
        [score >= 75, score >= 50],
        ["RED FLAG", "WATCH"],
        default="OK",
    )

    # ---------- Product-level ranking ----------
    product = base.groupby(
        ["sku_id", "sku_name", "category", "subcategory", "brand"],
        as_index=False,
    ).agg(
        forecast_units=("forecast_units", "sum"),
        actual_units=("actual_units", "sum"),
        forecast_revenue=("forecast_revenue", "sum"),
        forecast_profit=("forecast_profit", "sum"),
        value_at_stake=("value_at_stake", "sum"),
        stockout_value=("stockout_value", "sum"),
        overstock_value=("overstock_value", "sum"),
        max_risk_score=("risk_score", "max"),
    )

    product["profit_margin_pct"] = np.where(
        product["forecast_revenue"] > 0,
        product["forecast_profit"] / product["forecast_revenue"] * 100,
        0,
    )
    product["red_flag"] = np.select(
        [product["max_risk_score"] >= 75, product["max_risk_score"] >= 50],
        ["RED FLAG", "WATCH"],
        default="OK",
    )

    meta = {
        "categories": sorted(base["category"].unique().tolist()),
        "stores": sorted(base["store_id"].unique().tolist()),
        "risk_types": sorted(base["risk_type_clean"].unique().tolist()),
        "actions": sorted(base["action"].unique().tolist()),
        "skus": sorted(base["sku_id"].unique().tolist())[:10000],
    }

    print(f"DASHBOARD: built in {time.time() - started:.2f}s; {len(base):,} store-SKU rows")
    return base, product, meta


def initialize_dashboard_cache():
    """Build once during worker startup so the first browser request is fast."""
    global _DASHBOARD_CACHE, _DASHBOARD_BUILD_ERROR, _DASHBOARD_BUILD_SECONDS
    started = time.time()
    try:
        _DASHBOARD_CACHE = build_dashboard_data()
        _DASHBOARD_BUILD_ERROR = None
    except Exception as e:
        _DASHBOARD_CACHE = (pd.DataFrame(), pd.DataFrame(), {})
        _DASHBOARD_BUILD_ERROR = repr(e)
        print("DASHBOARD STARTUP ERROR:", repr(e))
    _DASHBOARD_BUILD_SECONDS = time.time() - started


# IMPORTANT: preload dashboard data at import time. This prevents the first
# user request from doing expensive pandas work through the Render proxy.
initialize_dashboard_cache()


@app.get("/dashboard_data")
def dashboard_data():
    """
    Fast dashboard endpoint. All heavy pandas work has already been cached.
    Filtering/ranking is done on the cached compact dataframe.
    """
    try:
        if _DASHBOARD_CACHE is None:
            return safe_json({
                "error": "Dashboard cache unavailable",
                "message": _DASHBOARD_BUILD_ERROR or "Unknown startup error",
                "data": [],
                "products": [],
                "meta": {},
            }, 200)

        base_all, product_all, meta = _DASHBOARD_CACHE

        if base_all.empty:
            return safe_json({
                "data": [],
                "products": [],
                "meta": meta,
                "kpis": {
                    "products": 0, "forecast_units": 0, "revenue": 0,
                    "profit": 0, "value_at_stake": 0, "red_flags": 0,
                    "reorder": 0, "markdown": 0
                },
                "count": 0,
                "warning": _DASHBOARD_BUILD_ERROR,
            })

        base = base_all
        product = product_all

        category = request.args.get("category", "").strip()
        sku = request.args.get("sku_id", "").strip()
        store = request.args.get("store_id", "").strip()
        risk_type = request.args.get("risk_type", "").strip().upper()
        action = request.args.get("action", "").strip().upper()
        red_flag = request.args.get("red_flag", "").strip().upper()
        search = request.args.get("search", "").strip().casefold()
        ranking = request.args.get("ranking", "revenue").strip().lower()

        # ---------- Filters ----------
        if category:
            base = base[base["category"].str.casefold() == category.casefold()]
            product = product[product["category"].str.casefold() == category.casefold()]
        if sku:
            base = base[base["sku_id"] == sku]
            product = product[product["sku_id"] == sku]
        if store:
            base = base[base["store_id"] == store]
        if risk_type:
            base = base[base["risk_type_clean"].str.upper() == risk_type]
        if action:
            base = base[base["action"].str.upper() == action]
        if red_flag:
            base = base[base["red_flag"].str.upper() == red_flag]
        if search:
            mask = (
                base["sku_name"].str.casefold().str.contains(search, regex=False, na=False)
                | base["brand"].str.casefold().str.contains(search, regex=False, na=False)
                | base["sku_id"].str.casefold().str.contains(search, regex=False, na=False)
            )
            base = base[mask]

            pmask = (
                product["sku_name"].str.casefold().str.contains(search, regex=False, na=False)
                | product["brand"].str.casefold().str.contains(search, regex=False, na=False)
                | product["sku_id"].str.casefold().str.contains(search, regex=False, na=False)
            )
            product = product[pmask]

        # Rebuild product ranking from filtered store-SKU data when a store/risk/action/
        # flag filter is used. This makes the ranking genuinely reflect the selection.
        if store or risk_type or action or red_flag or search:
            if base.empty:
                product = product.iloc[0:0]
            else:
                product = base.groupby(
                    ["sku_id", "sku_name", "category", "subcategory", "brand"],
                    as_index=False,
                ).agg(
                    forecast_units=("forecast_units", "sum"),
                    actual_units=("actual_units", "sum"),
                    forecast_revenue=("forecast_revenue", "sum"),
                    forecast_profit=("forecast_profit", "sum"),
                    value_at_stake=("value_at_stake", "sum"),
                    stockout_value=("stockout_value", "sum"),
                    overstock_value=("overstock_value", "sum"),
                    max_risk_score=("risk_score", "max"),
                )
                product["profit_margin_pct"] = np.where(
                    product["forecast_revenue"] > 0,
                    product["forecast_profit"] / product["forecast_revenue"] * 100,
                    0,
                )
                product["red_flag"] = np.select(
                    [product["max_risk_score"] >= 75, product["max_risk_score"] >= 50],
                    ["RED FLAG", "WATCH"],
                    default="OK",
                )

        # ---------- KPIs BEFORE table limiting ----------
        stockout_exposure = df.loc[
    df["risk_level"] == "STOCKOUT",
    "stockout_value_at_stake"
].sum()

overstock_exposure = df.loc[
    df["risk_level"] == "OVERSTOCK",
    "overstock_value_at_stake"
].sum()

total_value_at_stake = (
    stockout_exposure +
    overstock_exposure
) 
return
{
    "stockout_exposure": round(stockout_exposure, 2),
    "overstock_exposure": round(overstock_exposure, 2),
    "total_value_at_stake": round(total_value_at_stake, 2)
}
        
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

        risk_counts = {
            str(k): int(v)
            for k, v in base["risk_type_clean"].value_counts().items()
        }
        action_counts = {
            str(k): int(v)
            for k, v in base["action"].value_counts().items()
        }

        # ---------- Product ranking ----------
        rank_col = {
            "revenue": "forecast_revenue",
            "profit": "forecast_profit",
            "low_profit": "forecast_profit",
            "risk": "max_risk_score",
            "value": "value_at_stake",
        }.get(ranking, "forecast_revenue")

        ascending = ranking == "low_profit"
        product = product.sort_values(rank_col, ascending=ascending).head(50)

        # ---------- Decision table ----------
        n = limit_value(100, 1000)
        base_view = base.sort_values(
            ["red_flag", "risk_score", "value_at_stake"],
            ascending=[True, False, False],
        ).head(n)

        data_cols = [
            "store_id", "sku_id", "sku_name", "category", "subcategory", "brand",
            "forecast_units", "actual_units", "forecast_revenue", "forecast_profit",
            "profit_margin_pct", "stock_on_hand", "reorder_point", "safety_stock",
            "risk_type_clean", "risk_score", "priority", "reorder_qty",
            "shortage_units", "excess_units", "stockout_value", "overstock_value",
            "value_at_stake", "action", "red_flag",
        ]
        pcols = [
            "sku_id", "sku_name", "category", "subcategory", "brand",
            "forecast_units", "actual_units", "forecast_revenue",
            "forecast_profit", "profit_margin_pct", "value_at_stake",
            "stockout_value", "overstock_value", "max_risk_score", "red_flag",
        ]

        return safe_json({
            "data": records(base_view[[c for c in data_cols if c in base_view.columns]]),
            "products": records(product[[c for c in pcols if c in product.columns]]),
            "meta": meta,
            "kpis": kpis,
            "risk_counts": risk_counts,
            "action_counts": action_counts,
            "count": int(len(base)),
            "cache": True,
            "build_seconds": round(_DASHBOARD_BUILD_SECONDS, 3),
        })

    except Exception as e:
        print("DASHBOARD REQUEST ERROR:", repr(e))
        # Return valid JSON with 200 rather than letting Render return a blank 502.
        return safe_json({
            "error": "Dashboard request failed",
            "message": repr(e),
            "data": [],
            "products": [],
            "meta": {},
            "kpis": {},
            "count": 0,
        }, 200)


# ============================================================
# DASHBOARD UI
# ============================================================

DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NorthBay Foresight | AI Retail Control Tower</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
 --bg:rgb(7,12,24);--panel:rgb(15,23,42);--panel2:rgb(20,31,55);
 --text:rgb(241,245,249);--muted:rgb(148,163,184);--line:rgb(51,65,85);
 --cyan:rgb(34,211,238);--green:rgb(52,211,153);--red:rgb(248,113,113);
 --amber:rgb(251,191,36);--blue:rgb(96,165,250);--purple:rgb(167,139,250);
}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:
linear-gradient(135deg,rgb(5,10,22),rgb(12,20,38),rgb(8,15,31));color:var(--text)}
.wrap{max-width:1500px;margin:auto;padding:24px}
.hero{padding:28px;border:1px solid var(--line);border-radius:22px;
background:linear-gradient(135deg,rgb(17,27,49),rgb(22,35,62));
box-shadow:0 18px 55px rgba(0,0,0,.3)}
h1{margin:0;font-size:31px}.sub{margin-top:7px;color:var(--muted)}
.status{display:inline-flex;margin-top:15px;padding:7px 13px;border-radius:999px;
background:rgb(22,101,52);font-size:13px}.status.warn{background:rgb(146,64,14)}
.filters{margin-top:18px;display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}
.filters select,.filters input{width:100%;padding:12px;border-radius:11px;border:1px solid var(--line);
background:rgb(10,18,34);color:var(--text);outline:none}
.filters button{padding:12px;border:0;border-radius:11px;background:
linear-gradient(135deg,rgb(6,182,212),rgb(59,130,246));color:white;font-weight:800;cursor:pointer}
.btn2{background:rgb(51,65,85)!important}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}
.kpi{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:17px;padding:16px}.kpi small{color:var(--muted)}.kpi b{display:block;font-size:21px;margin-top:7px}
.money b{color:var(--cyan)}.good b{color:var(--green)}.danger b{color:var(--red)}
.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:14px}
.card{background:rgba(15,23,42,.9);border:1px solid var(--line);border-radius:18px;padding:16px;margin-bottom:14px}
.card h2{font-size:17px;margin:0 0 12px}.chart{height:330px}.wide{grid-column:1/-1}
.tablewrap{overflow:auto;max-height:440px;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:10px;border-bottom:1px solid rgb(30,41,59);white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:rgb(15,23,42);z-index:1;color:var(--muted)}
tr:hover{background:rgb(30,41,59)}.red{color:var(--red);font-weight:800}
.green{color:var(--green);font-weight:700}.amber{color:var(--amber);font-weight:800}
.note{color:var(--muted);font-size:12px;line-height:1.5}.error{color:var(--red);padding:18px}
@media(max-width:1100px){.filters{grid-template-columns:repeat(3,1fr)}.kpis{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}}
@media(max-width:650px){.wrap{padding:12px}.filters{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero">
<h1>NorthBay Foresight — AI Retail Control Tower</h1>
<div class="sub">Demand forecast • revenue & profit • inventory risk • recommended action</div>
<span id="status" class="status">Connecting…</span>

<div class="filters">
<select id="category"><option value="">All Categories</option></select>
<select id="sku"><option value="">All Product / SKU</option></select>
<select id="store"><option value="">All Stores</option></select>
<select id="risk"><option value="">All Risk Types</option></select>
<select id="action"><option value="">All Actions</option></select>
<select id="flag"><option value="">All Flags</option><option>RED FLAG</option><option>WATCH</option><option>OK</option></select>
<select id="ranking"><option value="revenue">Highest Revenue</option>
<option value="profit">Highest Profit</option><option value="low_profit">Lowest Profit</option>
<option value="risk">Highest Risk</option><option value="value">Highest Value at Stake</option></select>
<input id="search" placeholder="Search product name / brand / SKU">
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
<section class="card"><h2>Value at Stake — Top Products</h2><div id="valueChart" class="chart"></div></section>

<section class="card wide"><h2>🚨 Priority Decision Grid — What to do now</h2>
<div class="note">RED FLAG = immediate attention • REORDER = replenish stock • MARKDOWN / SELL NOW = accelerate excess inventory sales • WATCH = monitor • NO ACTION = stable.</div>
<div id="decisionTable" class="tablewrap"></div></section>

<section class="card wide"><h2>🏆 Product Ranking</h2>
<div class="note">Use the ranking filter to switch between highest revenue, highest profit, lowest profit, highest risk and highest value at stake.</div>
<div id="productTable" class="tablewrap"></div></section>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
const money=x=>'₹'+Number(x||0).toLocaleString('en-IN',{maximumFractionDigits:0});
const num=x=>Number(x||0).toLocaleString('en-IN',{maximumFractionDigits:1});

function fill(id,values){
 const el=$(id); el.innerHTML='<option value="">'+el.options[0]?.textContent+'</option>';
 (values||[]).forEach(v=>{
   const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o);
 });
}
function setK(id,v){$(id).textContent=v}

async function fetchJson(url,tries=3){
 for(let i=0;i<tries;i++){
   try{
     const res=await fetch(url,{cache:'no-store'});
     const text=await res.text();
     if(!res.ok) throw new Error('HTTP '+res.status+(text?' — '+text.slice(0,160):' — empty response'));
     if(!text.trim()) throw new Error('HTTP '+res.status+' — empty response');
     return JSON.parse(text);
   }catch(e){
     if(i===tries-1) throw e;
     await new Promise(r=>setTimeout(r,1000*(i+1)));
   }
 }
}

function plot(id,data,layout){
 if(typeof Plotly==='undefined')return;
 Plotly.newPlot(id,data,layout,{responsive:true,displayModeBar:false});
}

function tableHtml(rows,product){
 if(!rows||!rows.length)return '<div style="padding:20px;color:rgb(148,163,184)">No records match the selected filters.</div>';
 const cols=product?
 ['sku_id','sku_name','category','forecast_units','forecast_revenue','forecast_profit','profit_margin_pct','value_at_stake','max_risk_score','red_flag']:
 ['store_id','sku_id','sku_name','category','forecast_units','forecast_revenue','forecast_profit','risk_type_clean','risk_score','stock_on_hand','reorder_qty','action','value_at_stake','red_flag'];
 let h='<table><thead><tr>'+cols.map(c=>'<th>'+c.replaceAll('_',' ').toUpperCase()+'</th>').join('')+'</tr></thead><tbody>';
 rows.forEach(r=>{
   h+='<tr>'+cols.map(c=>{
     let v=r[c];
     if(['forecast_revenue','forecast_profit','value_at_stake'].includes(c))v=money(v);
     else if(['forecast_units','stock_on_hand','reorder_qty','shortage_units','excess_units'].includes(c))v=num(v);
     else if(['risk_score','max_risk_score','profit_margin_pct'].includes(c))v=num(v)+'%';
     const s=String(r[c]??'');
     const cls=s.includes('RED FLAG')||s.includes('URGENT')?'red':
               s.includes('MARKDOWN')||s.includes('SELL NOW')?'amber':
               s.includes('REORDER')?'red':'';
     return '<td class="'+cls+'">'+(v??'—')+'</td>';
   }).join('')+'</tr>';
 });
 return h+'</tbody></table>';
}

async function loadDashboard(){
 const q=new URLSearchParams();
 const map={category:'category',sku:'sku_id',store:'store_id',risk:'risk_type',action:'action',flag:'red_flag',ranking:'ranking',search:'search'};
 Object.entries(map).forEach(([id,key])=>{const v=$(id).value.trim();if(v)q.set(key,v)});
 q.set('limit','500');

 $('status').textContent='Loading dashboard…';$('status').className='status';

 try{
   const json=await fetchJson('/dashboard_data?'+q.toString());
   if(json.error)throw new Error(json.message||json.error);

   const k=json.kpis||{};
   setK('kProducts',num(k.products));setK('kUnits',num(k.forecast_units));
   setK('kRevenue',money(k.revenue));setK('kProfit',money(k.profit));
   setK('kStake',money(k.value_at_stake));setK('kFlags',num(k.red_flags));

   $('status').textContent='✓ Dashboard connected • '+num(json.count)+' filtered records';
   $('status').className='status';

   if(json.meta && !$('category').dataset.loaded){
     fill('category',json.meta.categories||[]);
     fill('sku',json.meta.skus||[]);
     fill('store',json.meta.stores||[]);
     fill('risk',json.meta.risk_types||[]);
     fill('action',json.meta.actions||[]);
     $('category').dataset.loaded='1';
   }

   const products=json.products||[];
   const labels=products.slice(0,12).map(x=>x.sku_id);

   plot('productChart',[
    {x:labels,y:products.slice(0,12).map(x=>x.forecast_revenue),type:'bar',name:'Revenue'},
    {x:labels,y:products.slice(0,12).map(x=>x.forecast_profit),type:'bar',name:'Profit'}
   ],{barmode:'group',margin:{t:10,l:65,r:10,b:70},paper_bgcolor:'transparent',
      plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},yaxis:{tickprefix:'₹'}});

   const rc=json.risk_counts||{};
   plot('riskChart',[{labels:Object.keys(rc),values:Object.values(rc),type:'pie',hole:.55}],
    {margin:{t:10,l:10,r:10,b:10},paper_bgcolor:'transparent',font:{color:'rgb(226,232,240)'}});

   const ac=json.action_counts||{};
   plot('actionChart',[{x:Object.keys(ac),y:Object.values(ac),type:'bar'}],
    {margin:{t:10,l:10,r:10,b:80},paper_bgcolor:'transparent',
     plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},xaxis:{tickangle:-25}});

   plot('valueChart',[{x:products.slice(0,10).map(x=>x.sku_id),
      y:products.slice(0,10).map(x=>x.value_at_stake),type:'bar'}],
    {margin:{t:10,l:65,r:10,b:70},paper_bgcolor:'transparent',
     plot_bgcolor:'transparent',font:{color:'rgb(226,232,240)'},yaxis:{tickprefix:'₹'}});

   $('decisionTable').innerHTML=tableHtml(json.data||[],false);
   $('productTable').innerHTML=tableHtml(products,true);

 }catch(e){
   $('status').textContent='⚠ '+e.message;$('status').className='status warn';
   $('decisionTable').innerHTML='<div class="error">Dashboard data could not be loaded: '+e.message+'</div>';
   $('productTable').innerHTML='';
 }
}

function resetFilters(){
 ['category','sku','store','risk','action','flag','ranking'].forEach(id=>$(id).selectedIndex=0);
 $('search').value='';
 loadDashboard();
}
loadDashboard();
</script>
</body>
</html>
"""

@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def global_error(e):
    print("UNHANDLED FLASK ERROR:", repr(e))
    return jsonify({"error": "Internal API error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

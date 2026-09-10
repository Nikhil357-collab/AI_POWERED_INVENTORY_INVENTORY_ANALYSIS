from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import os
import zipfile
import shutil
import time

# ============================================================
# NORTHBAY FORESIGHT
# DEPLOYMENT-SAFE FLASK API + DASHBOARD
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

_CACHE = {}
_DASHBOARD_CACHE = None
_DASHBOARD_BUILD_ERROR = None
_DASHBOARD_BUILD_SECONDS = 0.0

# ============================================================
# DATA DIRECTORY SETUP
# ============================================================

def setup_data_directory():
    if DATA_DIR.exists():
        found = sum((DATA_DIR / filename).is_file() for filename in REQUIRED_FILES)
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
            count = sum(filename in file_set for filename in REQUIRED_FILES)
            if count > best_count:
                best_count = count
                best_dir = Path(root)
        return best_dir

    print("DATA WARNING: data.zip not found")
    return DATA_DIR

DATA_DIR = setup_data_directory()

# ============================================================
# FILE MAP
# ============================================================

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

# ============================================================
# SAFE DATA HELPERS
# ============================================================

def load_csv(key):
    if key not in FILES: return pd.DataFrame()
    path = FILES[key]
    if not path.is_file(): return pd.DataFrame()
    if key in _CACHE: return _CACHE[key].copy()
    try:
        df = pd.read_csv(path, low_memory=False)
        _CACHE[key] = df.copy()
        return df
    except Exception as e:
        print(f"CSV ERROR [{key}]:", repr(e))
        return pd.DataFrame()

def num_col(df, col, default=0.0):
    if col not in df.columns: return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def first_col(df, candidates):
    for col in candidates:
        if col in df.columns: return col
    return None

def clean_value(value):
    if value is None or value is pd.NA: return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        try: return pd.Timestamp(value).strftime("%Y-%m-%d")
        except: return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)): return None
        return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value

def records(df):
    if df is None or df.empty: return []
    return [{str(k): clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def limit_value(default=100, maximum=5000):
    try: return max(1, min(int(request.args.get("limit", default)), maximum))
    except: return default

def safe_json(payload, status=200):
    try: return jsonify(payload), status
    except Exception as e: return jsonify({"error": "JSON error", "message": str(e)}), 500


# ============================================================
# ENDPOINTS RESTORED FOR STREAMLIT
# ============================================================

@app.get("/health")
def health():
    return safe_json({
        "status": "ok",
        "dashboard_cache": _DASHBOARD_CACHE is not None,
        "files": {key: bool(path.is_file()) for key, path in FILES.items()},
    })

def generic_csv_endpoint(key):
    df = load_csv(key)
    if df.empty:
        return safe_json({"data": [], "count": 0, "message": f"{key} file unavailable"})
    
    sku = request.args.get("sku_id", "").strip()
    store = request.args.get("store_id", "").strip()
    
    if sku and "sku_id" in df.columns: df = df[df["sku_id"].astype(str).eq(sku)]
    if store and "store_id" in df.columns: df = df[df["store_id"].astype(str).eq(store)]
        
    return safe_json({"data": records(df.head(limit_value(100, 5000))), "count": int(len(df))})

@app.get("/files")
def files_endpoint():
    return safe_json({k: {"exists": p.is_file()} for k, p in FILES.items()})

@app.get("/forecast")
def forecast_endpoint(): return generic_csv_endpoint("forecast")

@app.get("/risk")
def risk_endpoint(): return generic_csv_endpoint("risk")

@app.get("/reorder")
def reorder_endpoint(): return generic_csv_endpoint("reorder")

@app.get("/markdown")
def markdown_endpoint(): return generic_csv_endpoint("markdown")

@app.get("/decision")
def decision_endpoint(): return generic_csv_endpoint("decision")

@app.get("/metrics")
def metrics_endpoint(): return generic_csv_endpoint("metrics")

@app.get("/insights")
def insights_endpoint():
    df = load_csv("insights")
    if df.empty: return safe_json({})
    return safe_json({"data": records(df)})

@app.get("/sku/<sku_id>")
def sku_endpoint(sku_id):
    sku_id = str(sku_id)
    master, forecast, risk = load_csv("sku_master"), load_csv("forecast"), load_csv("risk")
    
    product = master[master["sku_id"].astype(str).eq(sku_id)] if not master.empty and "sku_id" in master.columns else pd.DataFrame()
    f_data = forecast[forecast["sku_id"].astype(str).eq(sku_id)].copy() if not forecast.empty and "sku_id" in forecast.columns else pd.DataFrame()
    r_data = risk[risk["sku_id"].astype(str).eq(sku_id)].copy() if not risk.empty and "sku_id" in risk.columns else pd.DataFrame()

    return safe_json({
        "sku_id": sku_id,
        "product": records(product.head(1)),
        "forecast": records(f_data.head(500)),
        "risk": records(r_data.head(100))
    })

# ============================================================
# FINANCIAL METRICS
# ============================================================

def calculate_financial_metrics(df):
    df = df.copy()
    numeric_columns = ["stockout_value", "overstock_value", "unit_price", "cost_price", "shortage_units", "excess_units", "stock_on_hand"]
    for col in numeric_columns:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    risk_col = first_col(df, ["risk_type", "risk_level"])
    df["risk_type_clean"] = df[risk_col].fillna("NORMAL").astype(str).str.upper().str.strip() if risk_col else "NORMAL"

    calc_stockout = df["shortage_units"] * df["unit_price"]
    df["stockout_value"] = np.where(df["stockout_value"] > 0, df["stockout_value"], calc_stockout)
    
    calc_overstock = df["excess_units"] * df["cost_price"]
    df["overstock_value"] = np.where(df["overstock_value"] > 0, df["overstock_value"], calc_overstock)

    is_stockout = df["risk_type_clean"].str.contains("STOCKOUT", na=False)
    is_overstock = df["risk_type_clean"].str.contains("OVERSTOCK", na=False)

    df["stockout_value"] = np.where(is_stockout, df["stockout_value"], 0.0)
    df["overstock_value"] = np.where(is_overstock, df["overstock_value"], 0.0)
    df["value_at_stake"] = df["stockout_value"] + df["overstock_value"]

    if "forecast_units" in df.columns:
        df["forecast_revenue"] = df["forecast_units"] * df["unit_price"]
        df["forecast_profit"] = df["forecast_units"] * (df["unit_price"] - df["cost_price"])
    else:
        df["forecast_revenue"] = 0.0
        df["forecast_profit"] = 0.0

    return df

# ============================================================
# DASHBOARD DATA ENGINE
# ============================================================

def build_dashboard_data():
    forecast, risk, master = load_csv("forecast"), load_csv("risk"), load_csv("sku_master")
    if forecast.empty and risk.empty: return (pd.DataFrame(), pd.DataFrame(), {})

    if forecast.empty:
        fagg = pd.DataFrame(columns=["store_id", "sku_id", "forecast_units", "actual_units"])
    else:
        f = forecast.copy()
        f["sku_id"] = f["sku_id"].astype(str) if "sku_id" in f.columns else "UNKNOWN"
        f["store_id"] = f["store_id"].astype(str) if "store_id" in f.columns else "ALL"
        f["prediction"] = num_col(f, "prediction")
        f["units_sold"] = num_col(f, "units_sold")
        fagg = f.groupby(["store_id", "sku_id"], as_index=False).agg(forecast_units=("prediction", "sum"), actual_units=("units_sold", "sum"))

    if risk.empty:
        r = pd.DataFrame(columns=["store_id", "sku_id", "risk_type_clean", "risk_score"])
    else:
        r = risk.copy()
        r["sku_id"] = r["sku_id"].astype(str) if "sku_id" in r.columns else "UNKNOWN"
        r["store_id"] = r["store_id"].astype(str) if "store_id" in r.columns else "ALL"
        risk_col = first_col(r, ["risk_type", "risk_level", "risk_type_clean"])
        r["risk_type_clean"] = r[risk_col].fillna("NORMAL").astype(str).str.upper().str.strip() if risk_col else "NORMAL"
        for col in ["risk_score", "stockout_value", "overstock_value", "reorder_qty", "shortage_units", "excess_units", "stock_on_hand"]:
            if col not in r.columns: r[col] = 0.0
            r[col] = num_col(r, col)
        r = r.sort_values("risk_score", ascending=False).drop_duplicates(["store_id", "sku_id"], keep="first")

    if fagg.empty: base = r.copy()
    elif r.empty: base = fagg.copy()
    else: base = fagg.merge(r, on=["store_id", "sku_id"], how="outer")

    if base.empty: return (pd.DataFrame(), pd.DataFrame(), {})
    base["sku_id"] = base["sku_id"].astype(str)
    base["store_id"] = base["store_id"].astype(str)

    if not master.empty and "sku_id" in master.columns:
        m = master.copy()
        m["sku_id"] = m["sku_id"].astype(str)
        keep = [col for col in ["sku_id", "sku_name", "category", "subcategory", "brand", "unit_price", "cost_price"] if col in m.columns]
        base = base.merge(m[keep].drop_duplicates("sku_id"), on="sku_id", how="left")

    for col in ["sku_name", "category", "subcategory", "brand"]:
        if col not in base.columns: base[col] = "Unknown"
        base[col] = base[col].fillna("Unknown").astype(str)

    base = calculate_financial_metrics(base)

    risk_type = base["risk_type_clean"].fillna("NORMAL").astype(str).str.upper().str.strip()
    score = base["risk_score"].clip(lower=0, upper=100)
    reorder_qty = num_col(base, "reorder_qty").clip(lower=0)
    excess = num_col(base, "excess_units").clip(lower=0)

    inferred = np.select(
        [
            (risk_type.str.contains("STOCKOUT") | (reorder_qty > 0)) & (score >= 75),
            (risk_type.str.contains("STOCKOUT") | (reorder_qty > 0)),
            (risk_type.str.contains("OVERSTOCK") | (excess > 0)) & (score >= 75),
            (risk_type.str.contains("OVERSTOCK") | (excess > 0)),
            score >= 50,
        ],
        ["URGENT REORDER", "REORDER", "URGENT MARKDOWN / SELL NOW", "MARKDOWN / SELL NOW", "WATCH CLOSELY"],
        default="NO ACTION"
    )
    
    if "recommended_action" in base.columns:
        existing = base["recommended_action"].fillna("").astype(str).str.upper().str.strip()
        valid_existing = existing.ne("") & ~existing.isin(["NAN", "NONE", "NO ACTION"])
        base["action"] = np.where(valid_existing, existing, inferred)
    else:
        base["action"] = inferred

    base["red_flag"] = np.select([score >= 75, score >= 50], ["RED FLAG", "WATCH"], default="OK")

    product = base.groupby(["sku_id", "sku_name", "category", "subcategory", "brand"], as_index=False).agg(
        forecast_units=("forecast_units", "sum"),
        forecast_revenue=("forecast_revenue", "sum"),
        forecast_profit=("forecast_profit", "sum"),
        value_at_stake=("value_at_stake", "sum"),
        max_risk_score=("risk_score", "max"),
    )
    
    meta = {
        "categories": sorted(base["category"].dropna().unique().tolist()),
        "stores": sorted(base["store_id"].dropna().unique().tolist()),
        "risk_types": sorted(base["risk_type_clean"].dropna().unique().tolist()),
        "actions": sorted(base["action"].dropna().unique().tolist()),
    }
    return (base, product, meta)

def initialize_dashboard_cache():
    global _DASHBOARD_CACHE, _DASHBOARD_BUILD_ERROR
    try:
        _DASHBOARD_CACHE = build_dashboard_data()
        _DASHBOARD_BUILD_ERROR = None
    except Exception as e:
        _DASHBOARD_CACHE = (pd.DataFrame(), pd.DataFrame(), {})
        _DASHBOARD_BUILD_ERROR = repr(e)

initialize_dashboard_cache()

@app.get("/dashboard_data")
def dashboard_data():
    try:
        if _DASHBOARD_CACHE is None: return safe_json({"error": "Cache unavailable", "message": _DASHBOARD_BUILD_ERROR})
        base_all, product_all, meta = _DASHBOARD_CACHE
        if base_all.empty: return safe_json({"data": [], "products": [], "meta": meta, "kpis": {}, "count": 0})

        base, product = base_all, product_all
        category = request.args.get("category", "").strip()
        sku = request.args.get("sku_id", "").strip()
        store = request.args.get("store_id", "").strip()
        ranking = request.args.get("ranking", "revenue").strip().lower()

        if category:
            base = base[base["category"].str.casefold().eq(category.casefold())]
            product = product[product["category"].str.casefold().eq(category.casefold())]
        if sku:
            base = base[base["sku_id"].eq(sku)]
            product = product[product["sku_id"].eq(sku)]
        if store:
            base = base[base["store_id"].eq(store)]

        kpis = {
            "products": int(base["sku_id"].nunique()),
            "forecast_units": float(base["forecast_units"].sum()),
            "revenue": float(base["forecast_revenue"].sum()),
            "profit": float(base["forecast_profit"].sum()),
            "value_at_stake": float(base["value_at_stake"].sum()),
            "red_flags": int((base["red_flag"] == "RED FLAG").sum()),
        }

        risk_counts = {str(k): int(v) for k, v in base["risk_type_clean"].value_counts().items()}
        action_counts = {str(k): int(v) for k, v in base["action"].value_counts().items()}

        rank_map = {"revenue": "forecast_revenue", "profit": "forecast_profit", "risk": "max_risk_score", "value": "value_at_stake"}
        rank_col = rank_map.get(ranking, "forecast_revenue")
        product = product.sort_values(rank_col, ascending=False).head(50)

        flag_priority = {"RED FLAG": 0, "WATCH": 1, "OK": 2}
        base_view = base.copy()
        
        # FIXED: Removed the bracket assignment causing the Pandas ChainedAssignmentError
        base_view = base_view.assign(_flag_priority=base_view["red_flag"].map(flag_priority).fillna(3))
        base_view = base_view.sort_values(["_flag_priority", "risk_score", "value_at_stake"], ascending=[True, False, False]).head(500)

        data_cols = ["store_id", "sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "risk_type_clean", "risk_score", "value_at_stake", "action", "red_flag"]
        prod_cols = ["sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "forecast_profit", "value_at_stake", "max_risk_score", "red_flag"]

        return safe_json({
            "data": records(base_view[[col for col in data_cols if col in base_view.columns]]),
            "products": records(product[[col for col in prod_cols if col in product.columns]]),
            "meta": meta, "kpis": kpis, "risk_counts": risk_counts, "action_counts": action_counts, "count": int(len(base)),
        })
    except Exception as e:
        return safe_json({"error": "Dashboard failed", "message": repr(e)}, 500)


@app.errorhandler(Exception)
def global_error(error):
    print("UNHANDLED FLASK ERROR:", repr(error))
    return jsonify({"error": "Internal API error", "message": str(error)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

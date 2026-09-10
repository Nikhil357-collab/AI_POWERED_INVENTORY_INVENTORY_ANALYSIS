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

        print(f"DATA: selected {best_dir} ({max(best_count, 0)}/{len(REQUIRED_FILES)} core files)")
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

print("=" * 70)
print("NORTHBAY FORESIGHT")
print("BASE_DIR :", BASE_DIR)
print("DATA_DIR :", DATA_DIR)
for key, path in FILES.items():
    print(f"{key:18s}: {'OK' if path.is_file() else 'MISSING'} -> {path.name}")
print("=" * 70)

# ============================================================
# SAFE DATA HELPERS
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
        print(f"CSV LOADED: {key} -> {len(df):,} rows")
        return df
    except Exception as e:
        print(f"CSV ERROR [{key}]:", repr(e))
        return pd.DataFrame()

def num_col(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def first_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def clean_value(value):
    if value is None or value is pd.NA: return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        try: return pd.Timestamp(value).strftime("%Y-%m-%d")
        except: return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)):
        try:
            if not np.isfinite(float(value)): return None
            return float(value)
        except: return None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value

def records(df):
    if df is None or df.empty: return []
    output = []
    for row in df.to_dict(orient="records"):
        output.append({str(key): clean_value(value) for key, value in row.items()})
    return output

def limit_value(default=100, maximum=1000):
    try: return max(1, min(int(request.args.get("limit", default)), maximum))
    except: return default

def safe_json(payload, status=200):
    try: return jsonify(payload), status
    except Exception as e: return jsonify({"error": "JSON error", "message": str(e)}), 500

# ============================================================
# HEALTH & FILE STATUS
# ============================================================

@app.get("/health")
def health():
    return safe_json({
        "status": "ok",
        "dashboard_cache": _DASHBOARD_CACHE is not None,
        "files": {key: bool(path.is_file()) for key, path in FILES.items()},
    })

# ============================================================
# FINANCIAL METRICS - WITH FIXES
# ============================================================

def calculate_financial_metrics(df):
    df = df.copy()
    
    numeric_columns = [
        "stockout_value", "overstock_value", "unit_price", "cost_price",
        "shortage_units", "excess_units", "stock_on_hand", "reorder_point", "safety_stock"
    ]
    
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Risk type cleaning
    risk_col = first_col(df, ["risk_type", "risk_level"])
    if risk_col:
        df["risk_type_clean"] = df[risk_col].fillna("NORMAL").astype(str).str.upper().str.strip()
    else:
        df["risk_type_clean"] = "NORMAL"

    # Auto-calculate values if the CSV doesn't have them
    calc_stockout = df["shortage_units"] * df["unit_price"]
    df["stockout_value"] = np.where(df["stockout_value"] > 0, df["stockout_value"], calc_stockout)
    
    calc_overstock = df["excess_units"] * df["cost_price"]
    df["overstock_value"] = np.where(df["overstock_value"] > 0, df["overstock_value"], calc_overstock)

    # Use 'contains' instead of 'eq' for fuzzy matching
    is_stockout = df["risk_type_clean"].str.contains("STOCKOUT", na=False)
    is_overstock = df["risk_type_clean"].str.contains("OVERSTOCK", na=False)

    df["stockout_value"] = np.where(is_stockout, df["stockout_value"], 0.0)
    df["overstock_value"] = np.where(is_overstock, df["overstock_value"], 0.0)

    # Final Value at Stake
    df["value_at_stake"] = df["stockout_value"] + df["overstock_value"]

    # Revenue
    if "forecast_units" in df.columns:
        df["forecast_revenue"] = df["forecast_units"] * df["unit_price"]
        df["forecast_profit"] = df["forecast_units"] * (df["unit_price"] - df["cost_price"])
    else:
        df["forecast_revenue"] = 0.0
        df["forecast_profit"] = 0.0

    df["profit_margin_pct"] = np.where(
        df["forecast_revenue"] > 0,
        (df["forecast_profit"] / df["forecast_revenue"] * 100), 0.0
    )
    return df

# ============================================================
# DASHBOARD DATA ENGINE
# ============================================================

def build_dashboard_data():
    started = time.time()
    forecast = load_csv("forecast")
    risk = load_csv("risk")
    master = load_csv("sku_master")

    if forecast.empty and risk.empty:
        return (pd.DataFrame(), pd.DataFrame(), {})

    # FORECAST AGGREGATION
    if forecast.empty:
        fagg = pd.DataFrame(columns=["store_id", "sku_id", "forecast_units", "actual_units"])
    else:
        f = forecast.copy()
        f["sku_id"] = f["sku_id"].astype(str) if "sku_id" in f.columns else "UNKNOWN"
        f["store_id"] = f["store_id"].astype(str) if "store_id" in f.columns else "ALL"
        f["prediction"] = num_col(f, "prediction")
        f["units_sold"] = num_col(f, "units_sold")

        fagg = f.groupby(["store_id", "sku_id"], as_index=False).agg(
            forecast_units=("prediction", "sum"),
            actual_units=("units_sold", "sum")
        )

    # RISK DATA
    if risk.empty:
        r = pd.DataFrame(columns=["store_id", "sku_id", "risk_type_clean", "risk_score"])
    else:
        r = risk.copy()
        r["sku_id"] = r["sku_id"].astype(str) if "sku_id" in r.columns else "UNKNOWN"
        r["store_id"] = r["store_id"].astype(str) if "store_id" in r.columns else "ALL"

        risk_column = first_col(r, ["risk_type", "risk_level", "risk_type_clean"])
        r["risk_type_clean"] = r[risk_column].fillna("NORMAL").astype(str).str.upper().str.strip() if risk_column else "NORMAL"

        for col in ["risk_score", "stockout_value", "overstock_value", "reorder_qty", "shortage_units", "excess_units", "stock_on_hand"]:
            if col not in r.columns: r[col] = 0.0
            r[col] = num_col(r, col)

        r = r.sort_values("risk_score", ascending=False).drop_duplicates(["store_id", "sku_id"], keep="first")

    # MERGE FORECAST + RISK
    if fagg.empty: base = r.copy()
    elif r.empty: base = fagg.copy()
    else: base = fagg.merge(r, on=["store_id", "sku_id"], how="outer")

    if base.empty: return (pd.DataFrame(), pd.DataFrame(), {})

    base["sku_id"] = base["sku_id"].astype(str)
    base["store_id"] = base["store_id"].astype(str)

    # SKU MASTER
    if not master.empty and "sku_id" in master.columns:
        m = master.copy()
        m["sku_id"] = m["sku_id"].astype(str)
        keep = [col for col in ["sku_id", "sku_name", "category", "subcategory", "brand", "unit_price", "cost_price"] if col in m.columns]
        m = m[keep].drop_duplicates("sku_id")
        base = base.merge(m, on="sku_id", how="left")

    for col in ["sku_name", "category", "subcategory", "brand"]:
        if col not in base.columns: base[col] = "Unknown"
        base[col] = base[col].fillna("Unknown").astype(str)

    # FINANCIAL METRICS
    base = calculate_financial_metrics(base)

    # DECISION ENGINE
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

    # PRODUCT RANKING
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
    
    product["profit_margin_pct"] = np.where(product["forecast_revenue"] > 0, (product["forecast_profit"] / product["forecast_revenue"] * 100), 0.0)
    product["red_flag"] = np.select([product["max_risk_score"] >= 75, product["max_risk_score"] >= 50], ["RED FLAG", "WATCH"], default="OK")

    meta = {
        "categories": sorted(base["category"].dropna().unique().tolist()),
        "stores": sorted(base["store_id"].dropna().unique().tolist()),
        "risk_types": sorted(base["risk_type_clean"].dropna().unique().tolist()),
        "actions": sorted(base["action"].dropna().unique().tolist()),
        "skus": sorted(base["sku_id"].dropna().unique().tolist())[:10000],
    }

    _DASHBOARD_BUILD_SECONDS = time.time() - started
    print(f"DASHBOARD BUILT: {_DASHBOARD_BUILD_SECONDS:.2f}s | {len(base):,} rows")
    return (base, product, meta)

# ============================================================
# DASHBOARD CACHE
# ============================================================

def initialize_dashboard_cache():
    global _DASHBOARD_CACHE, _DASHBOARD_BUILD_ERROR
    try:
        _DASHBOARD_CACHE = build_dashboard_data()
        _DASHBOARD_BUILD_ERROR = None
    except Exception as e:
        _DASHBOARD_CACHE = (pd.DataFrame(), pd.DataFrame(), {})
        _DASHBOARD_BUILD_ERROR = repr(e)

initialize_dashboard_cache()

# ============================================================
# DASHBOARD DATA API
# ============================================================

@app.get("/dashboard_data")
def dashboard_data():
    try:
        if _DASHBOARD_CACHE is None:
            return safe_json({"error": "Cache unavailable", "message": _DASHBOARD_BUILD_ERROR})

        base_all, product_all, meta = _DASHBOARD_CACHE
        if base_all.empty:
            return safe_json({"data": [], "products": [], "meta": meta, "kpis": {}, "count": 0})

        base = base_all
        product = product_all

        # Apply Filters
        category = request.args.get("category", "").strip()
        sku = request.args.get("sku_id", "").strip()
        store = request.args.get("store_id", "").strip()
        risk_type = request.args.get("risk_type", "").strip().upper()
        action = request.args.get("action", "").strip().upper()
        red_flag = request.args.get("red_flag", "").strip().upper()
        search = request.args.get("search", "").strip().casefold()
        ranking = request.args.get("ranking", "revenue").strip().lower()

        if category:
            base = base[base["category"].str.casefold().eq(category.casefold())]
            product = product[product["category"].str.casefold().eq(category.casefold())]
        if sku:
            base = base[base["sku_id"].eq(sku)]
            product = product[product["sku_id"].eq(sku)]
        if store:
            base = base[base["store_id"].eq(store)]
        if risk_type:
            base = base[base["risk_type_clean"].str.upper().eq(risk_type)]
        if action:
            base = base[base["action"].str.upper().eq(action)]
        if red_flag:
            base = base[base["red_flag"].str.upper().eq(red_flag)]
        if search:
            mask = base["sku_name"].str.casefold().str.contains(search, na=False) | base["sku_id"].str.casefold().str.contains(search, na=False)
            base = base[mask]
            p_mask = product["sku_name"].str.casefold().str.contains(search, na=False) | product["sku_id"].str.casefold().str.contains(search, na=False)
            product = product[p_mask]

        if store or risk_type or action or red_flag or search:
            if not base.empty:
                product = base.groupby(["sku_id", "sku_name", "category", "subcategory", "brand"], as_index=False).agg(
                    forecast_units=("forecast_units", "sum"),
                    actual_units=("actual_units", "sum"),
                    forecast_revenue=("forecast_revenue", "sum"),
                    forecast_profit=("forecast_profit", "sum"),
                    value_at_stake=("value_at_stake", "sum"),
                    max_risk_score=("risk_score", "max"),
                )
                product["profit_margin_pct"] = np.where(product["forecast_revenue"] > 0, (product["forecast_profit"] / product["forecast_revenue"] * 100), 0.0)
                product["red_flag"] = np.select([product["max_risk_score"] >= 75, product["max_risk_score"] >= 50], ["RED FLAG", "WATCH"], default="OK")

        # KPI Calculations
        kpis = {
            "products": int(base["sku_id"].nunique()),
            "store_sku": int(len(base)),
            "forecast_units": float(base["forecast_units"].sum()),
            "revenue": float(base["forecast_revenue"].sum()),
            "profit": float(base["forecast_profit"].sum()),
            "value_at_stake": float(base["value_at_stake"].sum()),
            "red_flags": int((base["red_flag"] == "RED FLAG").sum()),
        }

        risk_counts = {str(k): int(v) for k, v in base["risk_type_clean"].value_counts().items()}
        action_counts = {str(k): int(v) for k, v in base["action"].value_counts().items()}

        rank_map = {"revenue": "forecast_revenue", "profit": "forecast_profit", "low_profit": "forecast_profit", "risk": "max_risk_score", "value": "value_at_stake"}
        rank_col = rank_map.get(ranking, "forecast_revenue")
        product = product.sort_values(rank_col, ascending=(ranking == "low_profit")).head(50)

        # Decision Table prioritization
        flag_priority = {"RED FLAG": 0, "WATCH": 1, "OK": 2}
        base_view = base.copy()
        base_view["_flag_priority"] = base_view["red_flag"].map(flag_priority).fillna(3)
        base_view = base_view.sort_values(["_flag_priority", "risk_score", "value_at_stake"], ascending=[True, False, False]).head(500)

        data_cols = [
            "store_id", "sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "forecast_profit",
            "stock_on_hand", "risk_type_clean", "risk_score", "reorder_qty", "shortage_units", "excess_units",
            "value_at_stake", "action", "red_flag"
        ]
        
        prod_cols = [
            "sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "forecast_profit", 
            "profit_margin_pct", "value_at_stake", "max_risk_score", "red_flag"
        ]

        return safe_json({
            "data": records(base_view[[col for col in data_cols if col in base_view.columns]]),
            "products": records(product[[col for col in prod_cols if col in product.columns]]),
            "meta": meta,
            "kpis": kpis,
            "risk_counts": risk_counts,
            "action_counts": action_counts,
            "count": int(len(base)),
        })
    except Exception as e:
        return safe_json({"error": "Dashboard failed", "message": repr(e)}, 500)


# ============================================================
# DASHBOARD HTML (FULL UI RESTORED)
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
        :root {
            --bg: rgb(7,12,24); --panel: rgb(15,23,42); --panel2: rgb(20,31,55); --text: rgb(241,245,249);
            --muted: rgb(148,163,184); --line: rgb(51,65,85); --cyan: rgb(34,211,238);
            --green: rgb(52,211,153); --red: rgb(248,113,113); --amber: rgb(251,191,36);
        }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: linear-gradient(135deg, rgb(5,10,22), rgb(12,20,38), rgb(8,15,31)); color: var(--text); }
        .wrap { max-width: 1500px; margin: auto; padding: 24px; }
        .hero { padding: 28px; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(135deg, rgb(17,27,49), rgb(22,35,62)); box-shadow: 0 18px 55px rgba(0,0,0,.3); }
        h1 { margin: 0; font-size: 31px; }
        .sub { margin-top: 7px; color: var(--muted); }
        .status { display: inline-flex; margin-top: 15px; padding: 7px 13px; border-radius: 999px; background: rgb(22,101,52); font-size: 13px; }
        .status.warn { background: rgb(146,64,14); }
        .filters { margin-top: 18px; display: grid; grid-template-columns: repeat(4,minmax(150px,1fr)); gap: 10px; }
        .filters select, .filters input { width: 100%; padding: 12px; border-radius: 11px; border: 1px solid var(--line); background: rgb(10,18,34); color: var(--text); outline: none; }
        .filters button { padding: 12px; border: 0; border-radius: 11px; background: linear-gradient(135deg, rgb(6,182,212), rgb(59,130,246)); color: white; font-weight: 800; cursor: pointer; }
        .btn2 { background: rgb(51,65,85)!important; }
        .kpis { display: grid; grid-template-columns: repeat(6,1fr); gap: 12px; margin: 16px 0; }
        .kpi { background: linear-gradient(145deg, var(--panel), var(--panel2)); border: 1px solid var(--line); border-radius: 17px; padding: 16px; }
        .kpi small { color: var(--muted); }
        .kpi b { display: block; font-size: 21px; margin-top: 7px; }
        .money b { color: var(--cyan); }
        .good b { color: var(--green); }
        .danger b { color: var(--red); }
        .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }
        .card { background: rgba(15,23,42,.9); border: 1px solid var(--line); border-radius: 18px; padding: 16px; margin-bottom: 14px; }
        .card h2 { font-size: 17px; margin: 0 0 12px; }
        .chart { height: 330px; }
        .wide { grid-column: 1/-1; }
        .tablewrap { overflow: auto; max-height: 440px; border: 1px solid var(--line); border-radius: 12px; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 10px; border-bottom: 1px solid rgb(30,41,59); white-space: nowrap; text-align: left; }
        th { position: sticky; top: 0; background: rgb(15,23,42); z-index: 1; color: var(--muted); }
        tr:hover { background: rgb(30,41,59); }
        .red { color: var(--red); font-weight: 800; }
        .green { color: var(--green); font-weight: 700; }
        .amber { color: var(--amber); font-weight: 800; }
        .note { color: var(--muted); font-size: 12px; line-height: 1.5; }
        .error { color: var(--red); padding: 18px; }
        
        @media(max-width:1100px) { .filters { grid-template-columns: repeat(3,1fr); } .kpis { grid-template-columns: repeat(3,1fr); } .grid { grid-template-columns: 1fr; } }
        @media(max-width:650px) { .wrap { padding: 12px; } .filters { grid-template-columns: 1fr 1fr; } .kpis { grid-template-columns: 1fr 1fr; } }
    </style>
</head>
<body>
<div class="wrap">
    <section class="hero">
        <h1>NorthBay Foresight — AI Retail Control Tower</h1>
        <div class="sub">Demand forecast • Revenue & Profit • Inventory Risk • Recommended Action</div>
        <span id="status" class="status">Connecting…</span>
        
        <div class="filters">
            <select id="category"><option value="">All Categories</option></select>
            <select id="sku"><option value="">All Product / SKU</option></select>
            <select id="store"><option value="">All Stores</option></select>
            <select id="risk"><option value="">All Risk Types</option></select>
            <select id="action"><option value="">All Actions</option></select>
            <select id="flag">
                <option value="">All Flags</option>
                <option>RED FLAG</option>
                <option>WATCH</option>
                <option>OK</option>
            </select>
            <select id="ranking">
                <option value="revenue">Highest Revenue</option>
                <option value="profit">Highest Profit</option>
                <option value="low_profit">Lowest Profit</option>
                <option value="risk">Highest Risk</option>
                <option value="value">Highest Value at Stake</option>
            </select>
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
        <section class="card">
            <h2>Revenue vs Profit — Top Products</h2>
            <div id="productChart" class="chart"></div>
        </section>
        <section class="card">
            <h2>Risk Distribution</h2>
            <div id="riskChart" class="chart"></div>
        </section>
        <section class="card">
            <h2>Business Actions</h2>
            <div id="actionChart" class="chart"></div>
        </section>
        <section class="card">
            <h2>🚨 Value at Stake — Top Products</h2>
            <div id="valueChart" class="chart"></div>
        </section>
        <section class="card wide">
            <h2>🚨 Priority Decision Grid — What to do now</h2>
            <div class="note">RED FLAG = immediate attention • REORDER = replenish stock • MARKDOWN / SELL NOW = accelerate excess inventory sales • WATCH = monitor • NO ACTION = stable.</div>
            <div id="decisionTable" class="tablewrap"></div>
        </section>
        <section class="card wide">
            <h2>🏆 Product Ranking</h2>
            <div class="note">Use the ranking filter to switch between highest revenue, highest profit, lowest profit, highest risk and highest value at stake.</div>
            <div id="productTable" class="tablewrap"></div>
        </section>
    </div>
</div>

<script>
    const $ = id => document.getElementById(id);
    const money = x => '₹' + Number(x || 0).toLocaleString('en-IN', {maximumFractionDigits:0});
    const num = x => Number(x || 0).toLocaleString('en-IN', {maximumFractionDigits:1});

    function fill(id, values) {
        const el = $(id);
        const placeholder = el.options[0] ? el.options[0].textContent : '';
        el.innerHTML = '<option value="">' + placeholder + '</option>';
        (values || []).forEach(v => {
            const option = document.createElement('option');
            option.value = v; option.textContent = v; el.appendChild(option);
        });
    }

    function setK(id, value) { $(id).textContent = value; }

    async function fetchJson(url, tries = 3) {
        for(let i=0; i<tries; i++) {
            try {
                const response = await fetch(url, {cache: 'no-store'});
                const text = await response.text();
                if(!response.ok) throw new Error('HTTP ' + response.status);
                return JSON.parse(text);
            } catch(error) {
                if(i === tries - 1) throw error;
                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            }
        }
    }

    function plot(id, data, layout) {
        if(typeof Plotly === 'undefined') return;
        Plotly.newPlot(id, data, layout, {responsive:true, displayModeBar:false});
    }

    function tableHtml(rows, product) {
        if(!rows || !rows.length) return `<div style="padding:20px;color:rgb(148,163,184)">No records match filters.</div>`;
        
        const cols = product 
            ? ['sku_id', 'sku_name', 'category', 'forecast_units', 'forecast_revenue', 'forecast_profit', 'profit_margin_pct', 'value_at_stake', 'max_risk_score', 'red_flag']
            : ['store_id', 'sku_id', 'sku_name', 'category', 'forecast_units', 'forecast_revenue', 'forecast_profit', 'risk_type_clean', 'risk_score', 'stock_on_hand', 'reorder_qty', 'action', 'value_at_stake', 'red_flag'];

        let html = '<table><thead><tr>' + cols.map(c => '<th>' + c.replaceAll('_', ' ').toUpperCase() + '</th>').join('') + '</tr></thead><tbody>';
        
        rows.forEach(row => {
            html += '<tr>';
            cols.forEach(col => {
                let value = row[col];
                if(['forecast_revenue', 'forecast_profit', 'value_at_stake'].includes(col)) value = money(value);
                else if(['forecast_units', 'stock_on_hand', 'reorder_qty', 'shortage_units', 'excess_units', 'risk_score', 'max_risk_score'].includes(col)) value = num(value);
                else if(col === 'profit_margin_pct') value = num(value) + '%';
                
                const original = String(row[col] ?? '');
                let cls = '';
                if(original.includes('RED FLAG') || original.includes('URGENT') || original.includes('REORDER')) cls = 'red';
                else if(original.includes('MARKDOWN') || original.includes('SELL NOW')) cls = 'amber';
                
                html += '<td class="' + cls + '">' + (value ?? '—') + '</td>';
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        return html;
    }

    async function loadDashboard() {
        const query = new URLSearchParams();
        const map = {category: 'category', sku: 'sku_id', store: 'store_id', risk: 'risk_type', action: 'action', flag: 'red_flag', ranking: 'ranking', search: 'search'};
        
        Object.entries(map).forEach(([id,key]) => {
            const value = $(id).value.trim();
            if(value) query.set(key, value);
        });
        
        query.set('limit', '500');
        $('status').textContent = 'Loading dashboard…';
        $('status').className = 'status';

        try {
            const json = await fetchJson('/dashboard_data?' + query.toString());
            if(json.error) throw new Error(json.message || json.error);

            const k = json.kpis || {};
            setK('kProducts', num(k.products));
            setK('kUnits', num(k.forecast_units));
            setK('kRevenue', money(k.revenue));
            setK('kProfit', money(k.profit));
            setK('kStake', money(k.value_at_stake));
            setK('kFlags', num(k.red_flags));

            $('status').textContent = '✓ Dashboard connected • ' + num(json.count) + ' filtered records';
            $('status').className = 'status';

            if(json.meta && !$('category').dataset.loaded) {
                fill('category', json.meta.categories || []);
                fill('sku', json.meta.skus || []);
                fill('store', json.meta.stores || []);
                fill('risk', json.meta.risk_types || []);
                fill('action', json.meta.actions || []);
                $('category').dataset.loaded = '1';
            }

            const products = json.products || [];
            const labels = products.slice(0,12).map(x => String(x.sku_name || x.sku_id).substring(0, 15));
            
            plot('productChart', [
                {x: labels, y: products.slice(0,12).map(x => x.forecast_revenue), type: 'bar', name: 'Revenue'},
                {x: labels, y: products.slice(0,12).map(x => x.forecast_profit), type: 'bar', name: 'Profit'}
            ], {barmode: 'group', margin: {t:10, l:65, r:10, b:70}, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: 'rgb(226,232,240)'}, yaxis: {tickprefix:'₹'}});

            const riskCounts = json.risk_counts || {};
            plot('riskChart', [{labels: Object.keys(riskCounts), values: Object.values(riskCounts), type: 'pie', hole: .55}], {margin: {t:10, l:10, r:10, b:10}, paper_bgcolor: 'transparent', font: {color: 'rgb(226,232,240)'}});

            const actionCounts = json.action_counts || {};
            plot('actionChart', [{x: Object.keys(actionCounts), y: Object.values(actionCounts), type: 'bar'}], {margin: {t:10, l:10, r:10, b:80}, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: 'rgb(226,232,240)'}, xaxis: {tickangle:-25}});

            // Value at Stake Chart - Rendered with fixed red bars and proper names
            plot('valueChart', [{
                x: labels.slice(0,10), 
                y: products.slice(0,10).map(x => x.value_at_stake), 
                type: 'bar',
                marker: { color: 'rgb(248,113,113)' }
            }], {margin: {t:10, l:65, r:10, b:70}, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {color: 'rgb(226,232,240)'}, yaxis: {tickprefix:'₹'}});

            $('decisionTable').innerHTML = tableHtml(json.data || [], false);
            $('productTable').innerHTML = tableHtml(products, true);

        } catch(error) {
            $('status').textContent = '⚠ ' + error.message;
            $('status').className = 'status warn';
            $('decisionTable').innerHTML = '<div class="error">Dashboard data could not be loaded: ' + error.message + '</div>';
            $('productTable').innerHTML = '';
        }
    }

    function resetFilters() {
        ['category', 'sku', 'store', 'risk', 'action', 'flag'].forEach(id => $(id).selectedIndex = 0);
        $('ranking').selectedIndex = 0;
        $('search').value = '';
        loadDashboard();
    }

    loadDashboard();
</script>
</body>
</html>
"""

# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def global_error(error):
    print("UNHANDLED FLASK ERROR:", repr(error))
    return jsonify({
        "error": "Internal API error",
        "message": str(error),
    }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

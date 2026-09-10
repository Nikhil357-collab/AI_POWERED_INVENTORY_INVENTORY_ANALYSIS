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
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        try: return pd.Timestamp(value).strftime("%Y-%m-%d")
        except: return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        try:
            if not np.isfinite(float(value)): return None
            return float(value)
        except: return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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
# FINANCIAL METRICS - FIXED VALUE AT STAKE LOGIC
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

    # Use 'contains' instead of 'eq' to catch terms like "STOCKOUT RISK"
    is_stockout = df["risk_type_clean"].str.contains("STOCKOUT", na=False)
    is_overstock = df["risk_type_clean"].str.contains("OVERSTOCK", na=False)

    df["stockout_value"] = np.where(is_stockout, df["stockout_value"], 0.0)
    df["overstock_value"] = np.where(is_overstock, df["overstock_value"], 0.0)

    # Final Value at Stake (This guarantees the chart will have data)
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
        keep = [col for col in ["sku_id", "sku_name", "category", "brand", "unit_price", "cost_price"] if col in m.columns]
        m = m[keep].drop_duplicates("sku_id")
        base = base.merge(m, on="sku_id", how="left")

    for col in ["sku_name", "category", "brand"]:
        base[col] = base[col].fillna("Unknown").astype(str) if col in base.columns else "Unknown"

    for col in ["forecast_units", "actual_units", "risk_score", "stock_on_hand", "unit_price", "cost_price"]:
        base[col] = num_col(base, col)

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
        ["URGENT REORDER", "REORDER", "URGENT MARKDOWN", "MARKDOWN", "WATCH CLOSELY"],
        default="NO ACTION"
    )
    base["action"] = inferred
    base["red_flag"] = np.select([score >= 75, score >= 50], ["RED FLAG", "WATCH"], default="OK")

    # PRODUCT RANKING
    product = base.groupby(["sku_id", "sku_name", "category", "brand"], as_index=False).agg(
        forecast_units=("forecast_units", "sum"),
        forecast_revenue=("forecast_revenue", "sum"),
        forecast_profit=("forecast_profit", "sum"),
        value_at_stake=("value_at_stake", "sum"),
        max_risk_score=("risk_score", "max"),
    )
    
    product["profit_margin_pct"] = np.where(product["forecast_revenue"] > 0, (product["forecast_profit"] / product["forecast_revenue"] * 100), 0.0)
    product["red_flag"] = np.select([product["max_risk_score"] >= 75, product["max_risk_score"] >= 50], ["RED FLAG", "WATCH"], default="OK")

    meta = {
        "categories": sorted(base["category"].dropna().unique().tolist()),
        "stores": sorted(base["store_id"].dropna().unique().tolist()),
        "actions": sorted(base["action"].dropna().unique().tolist()),
    }
    
    _DASHBOARD_BUILD_SECONDS = time.time() - started
    print(f"DASHBOARD BUILT in {_DASHBOARD_BUILD_SECONDS:.2f}s | {len(base):,} rows")
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
            return safe_json({"data": [], "products": [], "count": 0})

        base, product = base_all, product_all

        # Apply Filters
        category = request.args.get("category", "").strip()
        search = request.args.get("search", "").strip().casefold()
        ranking = request.args.get("ranking", "value").strip().lower() # Default to Value to show it working

        if category:
            base = base[base["category"].str.casefold().eq(category.casefold())]
            product = product[product["category"].str.casefold().eq(category.casefold())]
            
        if search:
            mask = base["sku_name"].str.casefold().str.contains(search, na=False) | base["sku_id"].str.casefold().str.contains(search, na=False)
            base = base[mask]
            product = product[product["sku_name"].str.casefold().str.contains(search, na=False) | product["sku_id"].str.casefold().str.contains(search, na=False)]

        # Group if filters were applied
        if category or search:
            if not base.empty:
                product = base.groupby(["sku_id", "sku_name", "category", "brand"], as_index=False).agg(
                    forecast_units=("forecast_units", "sum"),
                    forecast_revenue=("forecast_revenue", "sum"),
                    forecast_profit=("forecast_profit", "sum"),
                    value_at_stake=("value_at_stake", "sum"),
                    max_risk_score=("risk_score", "max"),
                )

        kpis = {
            "products": int(base["sku_id"].nunique()),
            "forecast_units": float(base["forecast_units"].sum()),
            "revenue": float(base["forecast_revenue"].sum()),
            "profit": float(base["forecast_profit"].sum()),
            "value_at_stake": float(base["value_at_stake"].sum()),
            "red_flags": int((base["red_flag"] == "RED FLAG").sum()),
        }

        # Sorting based on UI selection
        rank_map = {"revenue": "forecast_revenue", "profit": "forecast_profit", "low_profit": "forecast_profit", "risk": "max_risk_score", "value": "value_at_stake"}
        rank_col = rank_map.get(ranking, "value_at_stake")
        product = product.sort_values(rank_col, ascending=(ranking == "low_profit")).head(50)

        data_cols = ["store_id", "sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "stock_on_hand", "risk_type_clean", "risk_score", "value_at_stake", "action", "red_flag"]
        prod_cols = ["sku_id", "sku_name", "category", "forecast_units", "forecast_revenue", "forecast_profit", "value_at_stake", "max_risk_score", "red_flag"]

        return safe_json({
            "data": records(base.head(500)[data_cols]),
            "products": records(product[prod_cols]),
            "meta": meta,
            "kpis": kpis,
            "count": int(len(base)),
        })
    except Exception as e:
        return safe_json({"error": "Dashboard failed", "message": repr(e)}, 500)


# ============================================================
# DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>NorthBay Foresight | Control Tower</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root { --bg: rgb(7,12,24); --panel: rgb(15,23,42); --text: rgb(241,245,249); --line: rgb(51,65,85); --cyan: rgb(34,211,238); --green: rgb(52,211,153); --red: rgb(248,113,113); --amber: rgb(251,191,36); }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: sans-serif; background: var(--bg); color: var(--text); }
        .wrap { max-width: 1500px; margin: auto; padding: 24px; }
        .hero { padding: 28px; border: 1px solid var(--line); border-radius: 22px; background: var(--panel); }
        h1 { margin: 0; font-size: 31px; }
        .filters { margin-top: 18px; display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
        .filters select, .filters input, .filters button { padding: 12px; border-radius: 11px; border: 1px solid var(--line); background: #000; color: #fff; width: 100%; }
        .filters button { background: rgb(34,211,238); color: #000; font-weight: bold; cursor: pointer; }
        .kpis { display: grid; grid-template-columns: repeat(6,1fr); gap: 12px; margin: 16px 0; }
        .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 17px; padding: 16px; }
        .kpi b { display: block; font-size: 21px; margin-top: 7px; }
        .danger b { color: var(--red); }
        .money b { color: var(--cyan); }
        .good b { color: var(--green); }
        .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }
        .card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 16px; margin-bottom:14px; }
        .chart { height: 330px; }
        .tablewrap { overflow: auto; max-height: 440px; border: 1px solid var(--line); }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 10px; border-bottom: 1px solid var(--line); text-align: left; }
        th { position: sticky; top: 0; background: var(--panel); }
        .red { color: var(--red); font-weight: bold; }
    </style>
</head>
<body>
<div class="wrap">
    <section class="hero">
        <h1>NorthBay Foresight</h1>
        <div class="filters">
            <select id="category"><option value="">All Categories</option></select>
            <select id="ranking">
                <option value="value">Highest Value at Stake</option>
                <option value="revenue">Highest Revenue</option>
                <option value="risk">Highest Risk</option>
            </select>
            <input id="search" placeholder="Search SKU...">
            <button onclick="loadDashboard()">Apply</button>
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
            <h2>Revenue vs Profit</h2>
            <div id="productChart" class="chart"></div>
        </section>
        <section class="card">
            <h2>🚨 Value at Stake (Top 10)</h2>
            <div id="valueChart" class="chart"></div>
        </section>
        <section class="card" style="grid-column: 1/-1;">
            <h2>Top Products Table</h2>
            <div id="productTable" class="tablewrap"></div>
        </section>
    </div>
</div>

<script>
    const $ = id => document.getElementById(id);
    const money = x => '₹' + Number(x||0).toLocaleString('en-IN');
    
    function fill(id, values) {
        const el = $(id);
        el.innerHTML = '<option value="">All</option>';
        (values||[]).forEach(v => el.appendChild(new Option(v, v)));
    }

    async function loadDashboard() {
        const url = `/dashboard_data?category=${$('category').value}&ranking=${$('ranking').value}&search=${$('search').value}`;
        const res = await fetch(url);
        const json = await res.json();
        
        $('kProducts').textContent = json.kpis.products;
        $('kUnits').textContent = json.kpis.forecast_units;
        $('kRevenue').textContent = money(json.kpis.revenue);
        $('kProfit').textContent = money(json.kpis.profit);
        $('kStake').textContent = money(json.kpis.value_at_stake);
        $('kFlags').textContent = json.kpis.red_flags;

        if(!$('category').dataset.loaded) {
            fill('category', json.meta.categories);
            $('category').dataset.loaded = '1';
        }

        const p = json.products.slice(0, 10);
        const labels = p.map(x => String(x.sku_name || x.sku_id).substring(0, 15));

        Plotly.newPlot('productChart', [
            {x: labels, y: p.map(x => x.forecast_revenue), type: 'bar', name: 'Revenue'},
            {x: labels, y: p.map(x => x.forecast_profit), type: 'bar', name: 'Profit'}
        ], {paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#fff'}});

        // Fixed Value At Stake Chart rendering with red bars
        Plotly.newPlot('valueChart', [
            {x: labels, y: p.map(x => x.value_at_stake), type: 'bar', marker: {color: 'rgb(248,113,113)'}}
        ], {paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#fff'}});

        let th = '<table><tr><th>SKU</th><th>Revenue</th><th>Profit</th><th>Value at Stake</th><th>Action</th></tr>';
        json.products.forEach(r => {
            th += `<tr><td>${r.sku_name || r.sku_id}</td><td>${money(r.forecast_revenue)}</td><td>${money(r.forecast_profit)}</td><td class="red">${money(r.value_at_stake)}</td><td class="red">${r.red_flag}</td></tr>`;
        });
        $('productTable').innerHTML = th + '</table>';
    }
    loadDashboard();
</script>
</body>
</html>
"""

@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

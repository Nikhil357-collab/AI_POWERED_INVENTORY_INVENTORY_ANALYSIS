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
            "/risk_summary", "/decision"
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
# FLASK DASHBOARD
# ============================================================

DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NORTHBAY FORESIGHT</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f5f7fb;color:#172033}
.wrap{max-width:1400px;margin:auto;padding:24px}
h1{margin-bottom:4px}.muted{color:#64748b}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}
.card{background:white;padding:18px;border-radius:14px;box-shadow:0 2px 10px #00000010}
.value{font-size:26px;font-weight:700;margin-top:7px}
.controls{background:white;padding:18px;border-radius:14px;margin-bottom:18px}
input,button{padding:10px;border:1px solid #d7dce5;border-radius:8px;margin:4px}
button{cursor:pointer;background:#172033;color:white}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{background:white;border-radius:14px;padding:18px;margin-bottom:18px}
.chart{height:390px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e8ebf0;text-align:left}
th{background:#f8fafc}
.status-ok{color:#16803c}.status-error{color:#b42318}
@media(max-width:900px){.cards,.grid{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.cards,.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<h1>📦 NORTHBAY FORESIGHT</h1>
<div class="muted">AI-powered demand forecasting and inventory decision intelligence</div>

<div class="controls">
<b>Filters</b><br>
<label>SKU ID:
<input id="sku" placeholder="e.g. SKU00001">
</label>
<label>Store ID:
<input id="store" placeholder="Optional">
</label>
<button onclick="loadAll()">Apply</button>
<button onclick="resetFilters()">Reset</button>
<span id="status" class="muted">Connecting...</span>
</div>

<div class="cards">
<div class="card">Forecast Records<div id="forecastCount" class="value">—</div></div>
<div class="card">Stockout Risk<div id="stockout" class="value">—</div></div>
<div class="card">Overstock Risk<div id="overstock" class="value">—</div></div>
<div class="card">Value at Stake<div id="value" class="value">—</div></div>
</div>

<div class="grid">
<div class="panel"><h3>💰 Value at Stake</h3><div id="valueChart" class="chart"></div></div>
<div class="panel"><h3>🚨 Risk Distribution</h3><div id="riskChart" class="chart"></div></div>
</div>

<div class="panel">
<h3>📈 Demand Forecast</h3>
<div id="forecastChart" class="chart"></div>
</div>

<div class="panel">
<h3>🔴 Reorder Priority</h3>
<div id="reorderTable">Loading...</div>
</div>

<div class="panel">
<h3>🟠 Markdown Priority</h3>
<div id="markdownTable">Loading...</div>
</div>

<div class="panel">
<h3>🚨 Risk Analysis</h3>
<div id="riskTable">Loading...</div>
</div>
</div>

<script>
function money(x){
    return "₹" + Number(x || 0).toLocaleString("en-IN",{maximumFractionDigits:0});
}

function params(){
    const p = new URLSearchParams();
    const sku = document.getElementById("sku").value.trim();
    const store = document.getElementById("store").value.trim();

    if(sku) p.set("sku_id",sku);
    if(store) p.set("store_id",store);

    return p.toString();
}

async function api(path){
    const q = params();
    const url = q ? path + (path.includes("?") ? "&" : "?") + q : path;
    const response = await fetch(url, {cache:"no-store"});

    if(!response.ok){
        throw new Error("HTTP " + response.status + " from " + path);
    }

    return await response.json();
}

function table(id, rows){
    const el = document.getElementById(id);

    if(!rows || !rows.length){
        el.innerHTML = "<p class='muted'>No records available.</p>";
        return;
    }

    const cols = Object.keys(rows[0]).slice(0,12);

    let html = "<table><thead><tr>";
    cols.forEach(c => html += "<th>" + c + "</th>");
    html += "</tr></thead><tbody>";

    rows.forEach(row => {
        html += "<tr>";

        cols.forEach(c => {
            let v = row[c];

            if(v === null || v === undefined) v = "";

            if(String(c).toLowerCase().includes("value")){
                v = money(v);
            }

            html += "<td>" + String(v) + "</td>";
        });

        html += "</tr>";
    });

    html += "</tbody></table>";
    el.innerHTML = html;
}

async function loadAll(){
    const status = document.getElementById("status");
    status.textContent = "Loading...";
    status.className = "muted";

    // allSettled prevents one failed endpoint from blanking the entire dashboard
    const results = await Promise.allSettled([
        api("/insights"),
        api("/forecast?limit=500"),
        api("/risk?limit=100"),
        api("/reorder?limit=20"),
        api("/markdown?limit=20")
    ]);

    const [insR, forecastR, riskR, reorderR, markdownR] = results;

    const ins = insR.status === "fulfilled" ? insR.value : {};
    const forecast = forecastR.status === "fulfilled" ? forecastR.value : {data:[]};
    const risk = riskR.status === "fulfilled" ? riskR.value : {data:[]};
    const reorder = reorderR.status === "fulfilled" ? reorderR.value : {data:[]};
    const markdown = markdownR.status === "fulfilled" ? markdownR.value : {data:[]};

    document.getElementById("forecastCount").textContent =
        Number(ins.forecast_records || 0).toLocaleString("en-IN");

    document.getElementById("stockout").textContent =
        Number(ins.stockout_count || 0).toLocaleString("en-IN");

    document.getElementById("overstock").textContent =
        Number(ins.overstock_count || 0).toLocaleString("en-IN");

    document.getElementById("value").textContent =
        money(ins.total_value_at_stake || 0);

    if(typeof Plotly !== "undefined"){
        Plotly.newPlot("valueChart",[{
            x:["Stockout","Overstock"],
            y:[Number(ins.stockout_value||0),Number(ins.overstock_value||0)],
            type:"bar"
        }],{
            margin:{t:10,l:60,r:20,b:50},
            yaxis:{tickprefix:"₹"}
        },{responsive:true});

        const rc = ins.risk_counts || {};

        Plotly.newPlot("riskChart",[{
            labels:Object.keys(rc),
            values:Object.values(rc),
            type:"pie"
        }],{
            margin:{t:10,l:10,r:10,b:10}
        },{responsive:true});
    }

    const fr = forecast.data || [];

    if(fr.length && fr[0].date && fr[0].prediction !== undefined){
        Plotly.newPlot("forecastChart",[{
            x:fr.map(x=>x.date),
            y:fr.map(x=>Number(x.prediction||0)),
            type:"scatter",
            mode:"lines+markers",
            name:"Forecast"
        }],{
            margin:{t:10,l:60,r:20,b:50},
            xaxis:{title:"Date"},
            yaxis:{title:"Predicted units"}
        },{responsive:true});
    }else{
        document.getElementById("forecastChart").innerHTML =
            "<p class='muted'>No forecast data available.</p>";
    }

    table("reorderTable", reorder.data || []);
    table("markdownTable", markdown.data || []);
    table("riskTable", risk.data || []);

    const failed = results.filter(x => x.status === "rejected");

    if(failed.length){
        status.textContent = "⚠ Dashboard loaded with " + failed.length + " endpoint warning(s)";
        status.className = "status-error";
    }else{
        status.textContent = "✓ API connected";
        status.className = "status-ok";
    }
}

function resetFilters(){
    document.getElementById("sku").value = "";
    document.getElementById("store").value = "";
    loadAll();
}

loadAll();
</script>
</body>
</html>
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

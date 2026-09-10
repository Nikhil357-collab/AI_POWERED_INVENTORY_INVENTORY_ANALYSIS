from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os

app = Flask(__name__)
CORS(app)

# ============================================================
# PATHS - WORKS LOCALLY AND ON RENDER
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Allow Render/local deployments where data may be in data/ directly.
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"

FILES = {
    "forecast": "xgboost_predictions.csv",
    "risk": "inventory_risk_scores.csv",
    "reorder": "reorder_priority_list.csv",
    "markdown": "markdown_priority_list.csv",
    "metrics": "xgmetrics.csv",
    "sku_master": "sku_master_clean.csv",
    "summary": "risk_summary.csv",
}

_CACHE = {}


def csv_path(key):
    return DATA_DIR / FILES[key]


def load_csv(key, required=False):
    """Load a CSV safely and cache it for fast Render responses."""
    path = csv_path(key)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.DataFrame()

    if key in _CACHE:
        return _CACHE[key].copy()

    try:
        df = pd.read_csv(path)
        _CACHE[key] = df
        return df.copy()
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def clean_records(df):
    """Convert pandas values into JSON-safe Python values."""
    if df is None or df.empty:
        return []

    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        else:
            out[col] = out[col].where(pd.notna(out[col]), None)

    records = out.to_dict(orient="records")

    # Handle numpy scalar types and non-finite floats.
    safe = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, float) and not math.isfinite(v):
                v = None
            clean[k] = v
        safe.append(clean)
    return safe


def apply_filters(df):
    """Common filters used by JSON endpoints."""
    if df.empty:
        return df

    sku = request.args.get("sku_id", "").strip()
    store = request.args.get("store_id", "").strip()
    category = request.args.get("category", "").strip()
    risk_type = request.args.get("risk_type", "").strip()

    if sku and "sku_id" in df.columns:
        df = df[df["sku_id"].astype(str) == sku]

    if store and "store_id" in df.columns:
        df = df[df["store_id"].astype(str) == store]

    if category and "category" in df.columns:
        df = df[df["category"].astype(str) == category]

    if risk_type and "risk_type" in df.columns:
        df = df[df["risk_type"].astype(str) == risk_type]

    return df


def limit_df(df, default=500, maximum=5000):
    try:
        limit = int(request.args.get("limit", default))
    except ValueError:
        limit = default
    limit = max(1, min(limit, maximum))
    return df.head(limit)


def number(df, col):
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/health")
def health():
    files = {key: csv_path(key).exists() for key in FILES}

    return jsonify({
        "service": "NORTHBAY FORESIGHT Flask Scoring API",
        "status": "healthy",
        "data_directory": str(DATA_DIR),
        "files": files,
        "endpoints": [
            "/", "/health", "/files", "/insights", "/metrics",
            "/forecast", "/risk", "/reorder", "/markdown",
            "/sku/<sku_id>"
        ],
    })


@app.get("/files")
def files():
    result = {}
    for key in FILES:
        path = csv_path(key)
        result[key] = {
            "file": FILES[key],
            "exists": path.exists(),
            "rows": int(len(load_csv(key))) if path.exists() else 0
        }
    return jsonify(result)


# ============================================================
# INSIGHTS
# ============================================================

@app.get("/insights")
def insights():
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
        if "risk_type" in risk.columns:
            counts = risk["risk_type"].astype(str).value_counts()
            result["risk_counts"] = {str(k): int(v) for k, v in counts.items()}
            result["stockout_count"] = int(counts.get("STOCKOUT", 0))
            result["overstock_count"] = int(counts.get("OVERSTOCK", 0))
            result["normal_count"] = int(counts.get("NORMAL", 0))

        # Support either old separate value columns or value_at_stake.
        result["stockout_value"] = number(risk, "stockout_value")
        result["overstock_value"] = number(risk, "overstock_value")

        if "value_at_stake" in risk.columns:
            result["total_value_at_stake"] = number(risk, "value_at_stake")
        else:
            result["total_value_at_stake"] = (
                result["stockout_value"] + result["overstock_value"]
            )

    return jsonify(result)


# ============================================================
# DATA API ROUTES
# ============================================================

@app.get("/metrics")
def metrics():
    df = load_csv("metrics")
    return jsonify({
        "data": clean_records(df),
        "count": int(len(df))
    })


@app.get("/forecast")
def forecast():
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


@app.get("/risk")
def risk():
    df = load_csv("risk")

    if df.empty:
        return jsonify({"data": [], "count": 0})

    df = apply_filters(df)

    # Highest risk first.
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
        df = df.sort_values("risk_score", ascending=False)

    df = limit_df(df, default=1000, maximum=5000)

    return jsonify({
        "data": clean_records(df),
        "count": int(len(df))
    })


@app.get("/reorder")
def reorder():
    df = load_csv("reorder")

    if df.empty:
        # Fallback: create reorder list directly from risk output.
        risk_df = load_csv("risk")
        if not risk_df.empty and "risk_type" in risk_df.columns:
            df = risk_df[
                risk_df["risk_type"].astype(str).str.upper() == "STOCKOUT"
            ].copy()

    df = apply_filters(df)

    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
        df = df.sort_values("risk_score", ascending=False)

    df = limit_df(df, default=100, maximum=5000)

    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/markdown")
def markdown():
    df = load_csv("markdown")

    if df.empty:
        risk_df = load_csv("risk")
        if not risk_df.empty and "risk_type" in risk_df.columns:
            df = risk_df[
                risk_df["risk_type"].astype(str).str.upper() == "OVERSTOCK"
            ].copy()

    df = apply_filters(df)

    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
        df = df.sort_values("risk_score", ascending=False)

    df = limit_df(df, default=100, maximum=5000)

    return jsonify({"data": clean_records(df), "count": int(len(df))})


@app.get("/sku/<sku_id>")
def sku(sku_id):
    sku_id = str(sku_id).strip()

    master = load_csv("sku_master")
    forecast_df = load_csv("forecast")
    risk_df = load_csv("risk")

    product = master[
        master["sku_id"].astype(str) == sku_id
    ].copy() if not master.empty and "sku_id" in master.columns else pd.DataFrame()

    f = forecast_df[
        forecast_df["sku_id"].astype(str) == sku_id
    ].copy() if not forecast_df.empty and "sku_id" in forecast_df.columns else pd.DataFrame()

    r = risk_df[
        risk_df["sku_id"].astype(str) == sku_id
    ].copy() if not risk_df.empty and "sku_id" in risk_df.columns else pd.DataFrame()

    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.sort_values("date")
    f = f.head(500)

    return jsonify({
        "sku_id": sku_id,
        "found": bool(not product.empty or not f.empty or not r.empty),
        "product": clean_records(product),
        "forecast": clean_records(f),
        "risk": clean_records(r),
    })


# ============================================================
# SIMPLE FLASK DASHBOARD
# This removes the need for Streamlit for submission.
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
input,select,button{padding:10px;border:1px solid #d7dce5;border-radius:8px;margin:4px}
button{cursor:pointer;background:#172033;color:white}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{background:white;border-radius:14px;padding:18px;margin-bottom:18px}
.chart{height:390px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e8ebf0;text-align:left}
th{background:#f8fafc}
.badge{font-weight:700}
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
    <input id="sku" placeholder="SKU00001">
  </label>
  <label>Store ID:
    <input id="store" placeholder="Optional">
  </label>
  <button onclick="loadAll()">Apply</button>
  <button onclick="resetFilters()">Reset</button>
  <span id="status" class="muted"></span>
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
  return "₹" + Number(x||0).toLocaleString("en-IN",{maximumFractionDigits:0});
}
function params(){
  const p = new URLSearchParams();
  const sku=document.getElementById("sku").value.trim();
  const store=document.getElementById("store").value.trim();
  if(sku)p.set("sku_id",sku);
  if(store)p.set("store_id",store);
  return p.toString();
}
async function api(path){
  const q=params();
  const url=q ? path+"?"+q : path;
  const r=await fetch(url);
  if(!r.ok) throw new Error("HTTP "+r.status);
  return await r.json();
}
function table(id, rows){
  if(!rows || !rows.length){
    document.getElementById(id).innerHTML="<p class='muted'>No records available.</p>";
    return;
  }
  const cols=Object.keys(rows[0]).slice(0,12);
  let h="<table><thead><tr>"+cols.map(c=>"<th>"+c+"</th>").join("")+"</tr></thead><tbody>";
  rows.forEach(row=>{
    h+="<tr>"+cols.map(c=>{
      let v=row[c];
      if(v===null||v===undefined)v="";
      if(String(c).toLowerCase().includes("value")) v=money(v);
      return "<td>"+v+"</td>";
    }).join("")+"</tr>";
  });
  h+="</tbody></table>";
  document.getElementById(id).innerHTML=h;
}
async function loadAll(){
  document.getElementById("status").textContent="Loading...";
  try{
    const [ins, forecast, risk, reorder, markdown] = await Promise.all([
      api("/insights"), api("/forecast?limit=500"), api("/risk?limit=100"),
      api("/reorder?limit=20"), api("/markdown?limit=20")
    ]);

    document.getElementById("forecastCount").textContent=Number(ins.forecast_records||0).toLocaleString("en-IN");
    document.getElementById("stockout").textContent=Number(ins.stockout_count||0).toLocaleString("en-IN");
    document.getElementById("overstock").textContent=Number(ins.overstock_count||0).toLocaleString("en-IN");
    document.getElementById("value").textContent=money(ins.total_value_at_stake);

    Plotly.newPlot("valueChart",[{
      x:["Stockout","Overstock"],
      y:[ins.stockout_value||0,ins.overstock_value||0],
      type:"bar"
    }],{margin:{t:10,l:60,r:20,b:50},yaxis:{tickprefix:"₹"}} ,{responsive:true});

    const rc=ins.risk_counts||{};
    Plotly.newPlot("riskChart",[{
      labels:Object.keys(rc),values:Object.values(rc),type:"pie"
    }],{margin:{t:10,l:10,r:10,b:10}},{responsive:true});

    const fr=forecast.data||[];
    if(fr.length && fr[0].date && fr[0].prediction!==undefined){
      Plotly.newPlot("forecastChart",[{
        x:fr.map(x=>x.date), y:fr.map(x=>Number(x.prediction||0)),
        type:"scatter", mode:"lines+markers", name:"Forecast"
      }],{margin:{t:10,l:60,r:20,b:50},xaxis:{title:"Date"},yaxis:{title:"Predicted units"}},
      {responsive:true});
    }else{
      document.getElementById("forecastChart").innerHTML="<p class='muted'>No forecast data available for the selected filter.</p>";
    }

    table("reorderTable",reorder.data||[]);
    table("markdownTable",markdown.data||[]);
    table("riskTable",risk.data||[]);
    document.getElementById("status").textContent="✓ API connected";
  }catch(e){
    document.getElementById("status").textContent="API error: "+e.message;
  }
}
function resetFilters(){
  document.getElementById("sku").value="";
  document.getElementById("store").value="";
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

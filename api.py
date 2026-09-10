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
# PATHS + DATA ZIP EXTRACTION
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
    """Find existing data or extract data.zip."""

    # --------------------------------------------------------
    # 1. Check normal data/processed directory
    # --------------------------------------------------------
    if DATA_DIR.exists():

        found = sum(
            (DATA_DIR / f).is_file()
            for f in REQUIRED_FILES
        )

        if found >= 3:
            print(f"Using existing data directory: {DATA_DIR}")
            print(f"Found {found}/{len(REQUIRED_FILES)} required files")
            return DATA_DIR

    # --------------------------------------------------------
    # 2. Extract data.zip
    # --------------------------------------------------------
    if ZIP_FILE.exists():

        print(f"Found data.zip: {ZIP_FILE}")

        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)

        EXTRACT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            with zipfile.ZipFile(
                ZIP_FILE,
                "r"
            ) as z:

                z.extractall(EXTRACT_DIR)

            print(
                f"data.zip extracted to: {EXTRACT_DIR}"
            )

        except Exception as e:

            print(
                f"ERROR extracting data.zip: {e}"
            )

            return DATA_DIR

        # ----------------------------------------------------
        # Search recursively
        # ----------------------------------------------------

        candidates = []

        for root, dirs, files in os.walk(EXTRACT_DIR):

            if "xgboost_predictions.csv" in files:

                candidate = Path(root)

                found = sum(
                    (candidate / f).is_file()
                    for f in REQUIRED_FILES
                )

                candidates.append(
                    (found, candidate)
                )

        if candidates:

            candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            found, candidate = candidates[0]

            print(
                f"Candidate data directory: {candidate}"
            )

            print(
                f"Found {found}/{len(REQUIRED_FILES)} required files"
            )

            if found >= 3:
                return candidate

        print(
            "WARNING: CSV files were not found "
            "in one common folder."
        )

        return EXTRACT_DIR

    print(
        f"WARNING: data.zip not found at {ZIP_FILE}"
    )

    return DATA_DIR


DATA_DIR = setup_data_directory()


# ============================================================
# FILE CONFIGURATION
# ============================================================

FILES = {

    "forecast":
        DATA_DIR / "xgboost_predictions.csv",

    "risk":
        DATA_DIR / "inventory_risk_scores.csv",

    "reorder":
        DATA_DIR / "reorder_priority_list.csv",

    "markdown":
        DATA_DIR / "markdown_priority_list.csv",

    "risk_summary":
        DATA_DIR / "risk_summary.csv",

    "sku_master":
        DATA_DIR / "sku_master_clean.csv",

    "insights":
        DATA_DIR / "business_insights.csv",

    "metrics":
        DATA_DIR / "xgmetrics.csv",

    "seasonal_metrics":
        DATA_DIR / "seasonal_naive_metrics.csv",

    "decision":
        DATA_DIR / "prioritised_decision_list.csv",
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

    if key not in FILES:

        print(
            f"Unknown data key: {key}"
        )

        return pd.DataFrame()

    path = FILES[key]

    if not path.is_file():

        print(
            f"File not found: {path}"
        )

        return pd.DataFrame()

    if key in _CACHE:

        return _CACHE[key].copy()

    try:

        df = pd.read_csv(
            path,
            low_memory=False
        )

        print(
            f"Loaded {key}: "
            f"{len(df):,} rows x "
            f"{len(df.columns)} columns"
        )

        _CACHE[key] = df.copy()

        return df

    except Exception as e:

        print(
            f"Failed to load {key}: {e}"
        )

        return pd.DataFrame()


# ============================================================
# JSON HELPERS
# ============================================================

def clean_records(df):

    if df is None or df.empty:
        return []

    out = df.copy()

    for col in out.columns:

        try:

            if pd.api.types.is_datetime64_any_dtype(
                out[col]
            ):

                out[col] = (
                    out[col]
                    .dt
                    .strftime("%Y-%m-%d")
                )

            else:

                out[col] = out[col].where(
                    pd.notna(out[col]),
                    None
                )

        except Exception:
            pass

    records = out.to_dict(
        orient="records"
    )

    safe = []

    for row in records:

        clean = {}

        for key, value in row.items():

            if value is None:

                clean[str(key)] = None

                continue

            if isinstance(
                value,
                np.integer
            ):

                value = int(value)

            elif isinstance(
                value,
                np.floating
            ):

                value = float(value)

            elif isinstance(
                value,
                np.bool_
            ):

                value = bool(value)

            elif isinstance(
                value,
                pd.Timestamp
            ):

                value = value.strftime(
                    "%Y-%m-%d"
                )

            if (
                isinstance(value, float)
                and not math.isfinite(value)
            ):

                value = None

            clean[str(key)] = value

        safe.append(clean)

    return safe


# ============================================================
# FILTERS
# ============================================================

def apply_filters(df):

    if df is None or df.empty:
        return df

    result = df.copy()

    sku = request.args.get(
        "sku_id",
        ""
    ).strip()

    store = request.args.get(
        "store_id",
        ""
    ).strip()

    category = request.args.get(
        "category",
        ""
    ).strip()

    risk_type = request.args.get(
        "risk_type",
        ""
    ).strip()

    risk_level = request.args.get(
        "risk_level",
        ""
    ).strip()

    if sku and "sku_id" in result.columns:

        result = result[
            result["sku_id"].astype(str)
            == sku
        ]

    if store and "store_id" in result.columns:

        result = result[
            result["store_id"].astype(str)
            == store
        ]

    if (
        category
        and "category" in result.columns
    ):

        result = result[
            result["category"].astype(str)
            == category
        ]

    # Support risk_type OR risk_level
    if risk_type:

        if "risk_type" in result.columns:

            result = result[
                result["risk_type"]
                .astype(str)
                .str.upper()
                == risk_type.upper()
            ]

        elif "risk_level" in result.columns:

            result = result[
                result["risk_level"]
                .astype(str)
                .str.upper()
                == risk_type.upper()
            ]

    if (
        risk_level
        and "risk_level" in result.columns
    ):

        result = result[
            result["risk_level"]
            .astype(str)
            .str.upper()
            == risk_level.upper()
        ]

    return result


def limit_df(
    df,
    default=500,
    maximum=5000
):

    try:

        limit = int(
            request.args.get(
                "limit",
                default
            )
        )

    except (
        TypeError,
        ValueError
    ):

        limit = default

    limit = max(
        1,
        min(
            limit,
            maximum
        )
    )

    return df.head(limit)


def number(df, col):

    if (
        df is None
        or df.empty
        or col not in df.columns
    ):

        return 0.0

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

    return float(values.sum())


def risk_column(df):

    if (
        df is not None
        and "risk_type" in df.columns
    ):

        return "risk_type"

    if (
        df is not None
        and "risk_level" in df.columns
    ):

        return "risk_level"

    return None


def get_risk_counts(df):

    if df is None or df.empty:
        return {}

    col = risk_column(df)

    if not col:
        return {}

    values = (
        df[col]
        .astype(str)
        .str.upper()
    )

    counts = values.value_counts()

    return {
        str(k): int(v)
        for k, v in counts.items()
    }


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_exception(error):

    print(
        f"API ERROR: "
        f"{type(error).__name__}: "
        f"{error}"
    )

    return jsonify({
        "status": "error",
        "error": type(error).__name__,
        "message": str(error)
    }), 500


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    file_status = {}

    for key, path in FILES.items():

        file_status[key] = {

            "exists":
                bool(path.is_file()),

            "file":
                path.name

        }

    return jsonify({

        "service":
            "NORTHBAY FORESIGHT Flask Scoring API",

        "status":
            "healthy",

        "data_directory":
            str(DATA_DIR),

        "files":
            file_status,

        "endpoints": [

            "/",
            "/health",
            "/files",
            "/insights",
            "/metrics",
            "/forecast",
            "/risk",
            "/reorder",
            "/markdown",
            "/sku/<sku_id>",
            "/seasonal_metrics",
            "/risk_summary",
            "/decision"

        ]

    })


# ============================================================
# FILE STATUS
# ============================================================

@app.get("/files")
def files():

    result = {}

    for key, path in FILES.items():

        exists = path.is_file()

        result[key] = {

            # IMPORTANT:
            # Convert Path -> string
            "file":
                path.name,

            "path":
                str(path),

            "exists":
                bool(exists),

            "rows":
                int(
                    len(load_csv(key))
                )
                if exists
                else 0

        }

    return jsonify({

        "status":
            "ok",

        "data_directory":
            str(DATA_DIR),

        "files":
            result

    })


# ============================================================
# INSIGHTS
# ============================================================

@app.get("/insights")
def insights():

    forecast = load_csv(
        "forecast"
    )

    risk = load_csv(
        "risk"
    )

    result = {

        "forecast_records":
            int(len(forecast)),

        "risk_records":
            int(len(risk)),

        "stockout_count":
            0,

        "overstock_count":
            0,

        "normal_count":
            0,

        "stockout_value":
            0.0,

        "overstock_value":
            0.0,

        "total_value_at_stake":
            0.0,

        "risk_counts":
            {},

        "percentage_stockout":
            0.0,

        "percentage_overstock":
            0.0

    }

    if not risk.empty:

        counts = get_risk_counts(
            risk
        )

        result["risk_counts"] = counts

        result["stockout_count"] = int(
            counts.get(
                "STOCKOUT",
                0
            )
        )

        result["overstock_count"] = int(
            counts.get(
                "OVERSTOCK",
                0
            )
        )

        result["normal_count"] = int(
            counts.get(
                "NORMAL",
                0
            )
        )

        result["stockout_value"] = number(
            risk,
            "stockout_value"
        )

        result["overstock_value"] = number(
            risk,
            "overstock_value"
        )

        if "value_at_stake" in risk.columns:

            result[
                "total_value_at_stake"
            ] = number(
                risk,
                "value_at_stake"
            )

        else:

            result[
                "total_value_at_stake"
            ] = (
                result["stockout_value"]
                +
                result["overstock_value"]
            )

        total = len(risk)

        if total > 0:

            result[
                "percentage_stockout"
            ] = round(
                result["stockout_count"]
                / total
                * 100,
                2
            )

            result[
                "percentage_overstock"
            ] = round(
                result["overstock_count"]
                / total
                * 100,
                2
            )

    return jsonify(result)


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def forecast():

    df = load_csv(
        "forecast"
    )

    if df.empty:

        return jsonify({

            "status":
                "ok",

            "data":
                [],

            "count":
                0

        })

    df = apply_filters(df)

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            "date"
        )

    df = limit_df(
        df,
        default=500,
        maximum=5000
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# RISK
# ============================================================

@app.get("/risk")
def risk():

    df = load_csv(
        "risk"
    )

    if df.empty:

        return jsonify({

            "status":
                "ok",

            "data":
                [],

            "count":
                0

        })

    df = apply_filters(df)

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        ).fillna(0)

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = limit_df(
        df,
        default=1000,
        maximum=5000
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# REORDER
# ============================================================

@app.get("/reorder")
def reorder():

    df = load_csv(
        "reorder"
    )

    if df.empty:

        risk_df = load_csv(
            "risk"
        )

        col = risk_column(
            risk_df
        )

        if (
            not risk_df.empty
            and col
        ):

            df = risk_df[
                risk_df[col]
                .astype(str)
                .str.upper()
                == "STOCKOUT"
            ].copy()

    df = apply_filters(df)

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        ).fillna(0)

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = limit_df(
        df,
        default=100,
        maximum=5000
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# MARKDOWN
# ============================================================

@app.get("/markdown")
def markdown():

    df = load_csv(
        "markdown"
    )

    if df.empty:

        risk_df = load_csv(
            "risk"
        )

        col = risk_column(
            risk_df
        )

        if (
            not risk_df.empty
            and col
        ):

            df = risk_df[
                risk_df[col]
                .astype(str)
                .str.upper()
                == "OVERSTOCK"
            ].copy()

    df = apply_filters(df)

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        ).fillna(0)

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = limit_df(
        df,
        default=100,
        maximum=5000
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# SKU
# ============================================================

@app.get("/sku/<sku_id>")
def sku(sku_id):

    sku_id = str(
        sku_id
    ).strip()

    master = load_csv(
        "sku_master"
    )

    forecast_df = load_csv(
        "forecast"
    )

    risk_df = load_csv(
        "risk"
    )

    product = pd.DataFrame()
    f = pd.DataFrame()
    r = pd.DataFrame()

    if (
        not master.empty
        and "sku_id" in master.columns
    ):

        product = master[
            master["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if (
        not forecast_df.empty
        and "sku_id" in forecast_df.columns
    ):

        f = forecast_df[
            forecast_df["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if (
        not risk_df.empty
        and "sku_id" in risk_df.columns
    ):

        r = risk_df[
            risk_df["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if "date" in f.columns:

        f["date"] = pd.to_datetime(
            f["date"],
            errors="coerce"
        )

        f = f.sort_values(
            "date"
        )

    f = f.head(500)

    return jsonify({

        "status":
            "ok",

        "sku_id":
            sku_id,

        "found":
            bool(
                not product.empty
                or not f.empty
                or not r.empty
            ),

        "product":
            clean_records(product),

        "forecast":
            clean_records(f),

        "risk":
            clean_records(r)

    })


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    df = load_csv(
        "metrics"
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# SEASONAL METRICS
# ============================================================

@app.get("/seasonal_metrics")
def seasonal_metrics():

    df = load_csv(
        "seasonal_metrics"
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# RISK SUMMARY
# ============================================================

@app.get("/risk_summary")
def risk_summary():

    df = load_csv(
        "risk_summary"
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# DECISION
# ============================================================

@app.get("/decision")
def decision():

    df = load_csv(
        "decision"
    )

    return jsonify({

        "status":
            "ok",

        "data":
            clean_records(df),

        "count":
            int(len(df))

    })


# ============================================================
# DASHBOARD
# ============================================================

DASHBOARD_HTML = r"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1">

<title>NORTHBAY FORESIGHT</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>

body{
    font-family:Arial,sans-serif;
    margin:0;
    background:#f5f7fb;
    color:#172033
}

.wrap{
    max-width:1400px;
    margin:auto;
    padding:24px
}

h1{
    margin-bottom:4px
}

.muted{
    color:#64748b
}

.cards{
    display:grid;
    grid-template-columns:
    repeat(4,1fr);
    gap:14px;
    margin:22px 0
}

.card{
    background:white;
    padding:18px;
    border-radius:14px;
    box-shadow:
    0 2px 10px #00000010
}

.value{
    font-size:26px;
    font-weight:700;
    margin-top:7px
}

.controls{
    background:white;
    padding:18px;
    border-radius:14px;
    margin-bottom:18px
}

input,
button{
    padding:10px;
    border:
    1px solid #d7dce5;
    border-radius:8px;
    margin:4px
}

button{
    cursor:pointer;
    background:#172033;
    color:white
}

.grid{
    display:grid;
    grid-template-columns:
    1fr 1fr;
    gap:18px
}

.panel{
    background:white;
    border-radius:14px;
    padding:18px;
    margin-bottom:18px
}

.chart{
    height:390px
}

table{
    width:100%;
    border-collapse:collapse;
    font-size:13px
}

th,
td{
    padding:9px;
    border-bottom:
    1px solid #e8ebf0;
    text-align:left
}

th{
    background:#f8fafc
}

@media(max-width:900px){

    .cards,
    .grid{
        grid-template-columns:
        1fr 1fr
    }

}

@media(max-width:600px){

    .cards,
    .grid{
        grid-template-columns:
        1fr
    }

}

</style>

</head>

<body>

<div class="wrap">

<h1>
📦 NORTHBAY FORESIGHT
</h1>

<div class="muted">
AI-powered demand forecasting
and inventory decision intelligence
</div>


<div class="controls">

<b>Filters</b>

<br>

<label>
SKU ID:

<input
id="sku"
placeholder="SKU00001">
</label>

<label>
Store ID:

<input
id="store"
placeholder="Optional">
</label>

<button onclick="loadAll()">
Apply
</button>

<button onclick="resetFilters()">
Reset
</button>

<span
id="status"
class="muted">
</span>

</div>


<div class="cards">

<div class="card">

Forecast Records

<div
id="forecastCount"
class="value">
—
</div>

</div>


<div class="card">

Stockout Risk

<div
id="stockout"
class="value">
—
</div>

</div>


<div class="card">

Overstock Risk

<div
id="overstock"
class="value">
—
</div>

</div>


<div class="card">

Value at Stake

<div
id="value"
class="value">
—
</div>

</div>

</div>


<div class="grid">


<div class="panel">

<h3>
💰 Value at Stake
</h3>

<div
id="valueChart"
class="chart">
</div>

</div>


<div class="panel">

<h3>
🚨 Risk Distribution
</h3>

<div
id="riskChart"
class="chart">
</div>

</div>

</div>


<div class="panel">

<h3>
📈 Demand Forecast
</h3>

<div
id="forecastChart"
class="chart">
</div>

</div>


<div class="panel">

<h3>
🔴 Reorder Priority
</h3>

<div
id="reorderTable">
Loading...
</div>

</div>


<div class="panel">

<h3>
🟠 Markdown Priority
</h3>

<div
id="markdownTable">
Loading...
</div>

</div>


<div class="panel">

<h3>
🚨 Risk Analysis
</h3>

<div
id="riskTable">
Loading...
</div>

</div>

</div>


<script>

function money(x){

    return "₹" +
    Number(x || 0)
    .toLocaleString(
        "en-IN",
        {
            maximumFractionDigits:0
        }
    );

}


function params(){

    const p =
    new URLSearchParams();

    const sku =
    document
    .getElementById("sku")
    .value
    .trim();

    const store =
    document
    .getElementById("store")
    .value
    .trim();

    if(sku){
        p.set(
            "sku_id",
            sku
        );
    }

    if(store){
        p.set(
            "store_id",
            store
        );
    }

    return p.toString();

}


async function api(path){

    const q =
    params();

    const url =
    q
    ? path +
      (path.includes("?")
       ? "&"
       : "?") +
      q
    : path;

    const response =
    await fetch(url);

    if(!response.ok){

        let message =
        "HTTP " +
        response.status;

        try{

            const body =
            await response.json();

            if(body.message){

                message +=
                ": " +
                body.message;

            }

        }catch(e){}

        throw new Error(
            message
        );

    }

    return await response.json();

}


function escapeHtml(value){

    return String(
        value ?? ""
    )
    .replaceAll(
        "&",
        "&amp;"
    )
    .replaceAll(
        "<",
        "&lt;"
    )
    .replaceAll(
        ">",
        "&gt;"
    )
    .replaceAll(
        '"',
        "&quot;"
    )
    .replaceAll(
        "'",
        "&#039;"
    );

}


function table(
    id,
    rows
){

    if(
        !rows ||
        !rows.length
    ){

        document
        .getElementById(id)
        .innerHTML =
        "<p class='muted'>" +
        "No records available." +
        "</p>";

        return;

    }


    const cols =
    Object.keys(
        rows[0]
    ).slice(0,12);


    let html =
    "<table><thead><tr>" +

    cols
    .map(
        c =>
        "<th>" +
        escapeHtml(c) +
        "</th>"
    )
    .join("") +

    "</tr></thead><tbody>";


    rows.forEach(
        row => {

            html += "<tr>";

            cols.forEach(
                c => {

                    let value =
                    row[c];

                    if(
                        value === null ||
                        value === undefined
                    ){

                        value = "";

                    }

                    if(
                        String(c)
                        .toLowerCase()
                        .includes("value")
                    ){

                        value =
                        money(value);

                    }

                    html +=
                    "<td>" +
                    escapeHtml(value) +
                    "</td>";

                }
            );

            html += "</tr>";

        }
    );


    html +=
    "</tbody></table>";


    document
    .getElementById(id)
    .innerHTML = html;

}


async function loadAll(){

    document
    .getElementById("status")
    .textContent =
    "Loading...";


    try{

        const [
            ins,
            forecast,
            risk,
            reorder,
            markdown
        ] = await Promise.all([

            api("/insights"),

            api(
                "/forecast?limit=500"
            ),

            api(
                "/risk?limit=100"
            ),

            api(
                "/reorder?limit=20"
            ),

            api(
                "/markdown?limit=20"
            )

        ]);


        document
        .getElementById(
            "forecastCount"
        )
        .textContent =
        Number(
            ins.forecast_records || 0
        )
        .toLocaleString(
            "en-IN"
        );


        document
        .getElementById(
            "stockout"
        )
        .textContent =
        Number(
            ins.stockout_count || 0
        )
        .toLocaleString(
            "en-IN"
        );


        document
        .getElementById(
            "overstock"
        )
        .textContent =
        Number(
            ins.overstock_count || 0
        )
        .toLocaleString(
            "en-IN"
        );


        document
        .getElementById(
            "value"
        )
        .textContent =
        money(
            ins.total_value_at_stake
        );


        if(
            typeof Plotly !==
            "undefined"
        ){

            Plotly.newPlot(

                "valueChart",

                [{

                    x:[
                        "Stockout",
                        "Overstock"
                    ],

                    y:[
                        ins.stockout_value || 0,
                        ins.overstock_value || 0
                    ],

                    type:"bar"

                }],

                {

                    margin:{
                        t:10,
                        l:60,
                        r:20,
                        b:50
                    },

                    yaxis:{
                        tickprefix:"₹"
                    }

                },

                {
                    responsive:true
                }

            );


            const rc =
            ins.risk_counts || {};


            Plotly.newPlot(

                "riskChart",

                [{

                    labels:
                    Object.keys(rc),

                    values:
                    Object.values(rc),

                    type:"pie"

                }],

                {
                    margin:{
                        t:10,
                        l:10,
                        r:10,
                        b:10
                    }
                },

                {
                    responsive:true
                }

            );


            const fr =
            forecast.data || [];


            if(
                fr.length &&
                fr[0].date &&
                fr[0].prediction !==
                undefined
            ){

                const traces = [{

                    x:
                    fr.map(
                        x => x.date
                    ),

                    y:
                    fr.map(
                        x =>
                        Number(
                            x.prediction || 0
                        )
                    ),

                    type:"scatter",

                    mode:
                    "lines+markers",

                    name:
                    "Forecast"

                }];


                if(
                    fr[0].units_sold !==
                    undefined
                ){

                    traces.push({

                        x:
                        fr.map(
                            x => x.date
                        ),

                        y:
                        fr.map(
                            x =>
                            Number(
                                x.units_sold || 0
                            )
                        ),

                        type:"scatter",

                        mode:
                        "lines",

                        name:
                        "Actual"

                    });

                }


                Plotly.newPlot(

                    "forecastChart",

                    traces,

                    {

                        margin:{
                            t:10,
                            l:60,
                            r:20,
                            b:50
                        },

                        xaxis:{
                            title:"Date"
                        },

                        yaxis:{
                            title:"Units"
                        }

                    },

                    {
                        responsive:true
                    }

                );

            }else{

                document
                .getElementById(
                    "forecastChart"
                )
                .innerHTML =
                "<p class='muted'>" +
                "No forecast data available." +
                "</p>";

            }

        }


        table(
            "reorderTable",
            reorder.data || []
        );


        table(
            "markdownTable",
            markdown.data || []
        );


        table(
            "riskTable",
            risk.data || []
        );


        document
        .getElementById(
            "status"
        )
        .textContent =
        "✓ API connected";


    }catch(error){

        console.error(error);

        document
        .getElementById(
            "status"
        )
        .textContent =
        "API error: " +
        error.message;

    }

}


function resetFilters(){

    document
    .getElementById("sku")
    .value = "";

    document
    .getElementById("store")
    .value = "";

    loadAll();

}


loadAll();

</script>

</body>

</html>
"""


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.get("/")
def dashboard():

    return render_template_string(
        DASHBOARD_HTML
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

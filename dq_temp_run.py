import json
from datetime import datetime
from pathlib import Path
import plotly
from plotly.offline import get_plotlyjs

# Verify Plotly version
print(f"Using Plotly version: {plotly.__version__}")

# Map your 4 sensor keys to display labels and JSON file paths
SENSOR_FILES = {
    "Sensor_1": {"label": "Sensor 1", "path": Path("sensor_1.json")},
    "Sensor_2": {"label": "Sensor 2", "path": Path("sensor_2.json")},
    "Sensor_3": {"label": "Sensor 3", "path": Path("sensor_3.json")},
    "Sensor_4": {"label": "Sensor 4", "path": Path("sensor_4.json")},
}

TEMPLATE_PATH = Path("template.html")
OUTPUT_HTML_PATH = Path("data_quality_dashboard.html")


def parse_sensor_json(file_path: Path) -> dict:
    """
    Parses JSON structured as:
    {
      "Coverage": {
        "Table_AN": {"YYYY-MM-DD": float, ...},
        "Table_DI": {"YYYY-MM-DD": float, ...}
      }
    }
    """
    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    coverage = raw.get("Coverage", {})
    table_an = coverage.get("Table_AN", {})
    table_di = coverage.get("Table_DI", {})

    # Union of all sorted YYYY-MM-DD dates across both channels
    all_dates = sorted(set(table_an.keys()) | set(table_di.keys()))

    an_values = [
        round(float(table_an[d]), 2) if d in table_an and table_an[d] is not None else None
        for d in all_dates
    ]
    di_values = [
        round(float(table_di[d]), 2) if d in table_di and table_di[d] is not None else None
        for d in all_dates
    ]

    return {
        "dates": all_dates,
        "an": an_values,
        "di": di_values,
    }


def build_standalone_html():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH.resolve()}")

    sensors_payload = {}
    for sensor_key, meta in SENSOR_FILES.items():
        json_path = meta["path"]
        if not json_path.exists():
            print(f"[WARNING] Missing {json_path}, skipping {sensor_key}...")
            continue

        parsed = parse_sensor_json(json_path)
        sensors_payload[sensor_key] = {
            "label": meta["label"],
            "dates": parsed["dates"],
            "an": parsed["an"],
            "di": parsed["di"],
        }
        print(f"Loaded {sensor_key}: {len(parsed['dates'])} daily records")

    if not sensors_payload:
        raise RuntimeError("No valid sensor JSON files were loaded.")

    app_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "plotly_version": plotly.__version__,
        "sensors": sensors_payload,
    }

    # Extract full offline Plotly 5.6.0 JS bundle (zero CDN dependency)
    plotly_js_bundle = get_plotlyjs()

    template_html = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Inject Plotly JS and JSON payload
    final_html = template_html.replace(
        "/* __PLOTLY_JS_PLACEHOLDER__ */", plotly_js_bundle
    ).replace(
        "/* __SENSOR_DATA_PLACEHOLDER__ */ {}", json.dumps(app_data, separators=(",", ":"))
    )

    OUTPUT_HTML_PATH.write_text(final_html, encoding="utf-8")
    size_mb = OUTPUT_HTML_PATH.stat().st_size / (1024 * 1024)
    print(f"Successfully generated {OUTPUT_HTML_PATH.resolve()} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    build_standalone_html()

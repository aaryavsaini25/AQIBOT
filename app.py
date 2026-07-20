from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed on the server, we ship the image as base64 instead
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__, static_folder=".", static_url_path="")

DATA_URL = "https://raw.githubusercontent.com/adityarc19/aqi-india/refs/heads/main/city_day.csv"
# Load once at startup instead of on every request (the file doesn't change per-request)
FULL_DATA = pd.read_csv(DATA_URL)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/submit", methods=["POST"])
def submit():
    req = request.get_json(force=True)
    city = (req.get("city") or "").strip()
    year_raw = (req.get("year") or "").strip()

    if not city:
        return jsonify({"message": "Please enter a city name."})

    try:
        future_year = int(year_raw)
    except (TypeError, ValueError):
        return jsonify({"message": "Please enter a valid year (a whole number)."})

    # ---- Same logic as the original script, just using `city`/`future_year`
    # ---- instead of input() ----
    data = FULL_DATA[FULL_DATA["City"] == city].copy()

    if data.empty:
        return jsonify({
            "message": f"No AQI data found for '{city}'. Please check the spelling, "
                       f"punctuation and capitalization."
        })

    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data = data.dropna(subset=["AQI", "AQI_Bucket"])
    data = data.drop(columns=[
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
        "Benzene", "Toluene", "Xylene"
    ])

    yearly_average = data.groupby("Year")["AQI"].mean().reset_index()

    if yearly_average.empty:
        return jsonify({"message": f"Not enough AQI data available for '{city}' to build a prediction."})

    x = yearly_average[["Year"]]
    y_log = np.log(yearly_average["AQI"])
    model = LinearRegression()
    model.fit(x, y_log)

    future_data = pd.DataFrame({"Year": [future_year]})
    predicted_log_AQI = model.predict(future_data)[0]
    predicted_AQI = np.exp(predicted_log_AQI)

    limit = 2031
    drop = 2010

    text_lines = []
    if future_year > limit or future_year < drop:
        text_lines.append(f"Please do not enter any year that exceeds {limit} or drops below {drop}.")
    else:
        text_lines.append(f"The predicted AQI for {city} for {future_year} is {predicted_AQI:.2f}.")

    text_lines.append(f"Here is a graph for {city}'s historic AQI data and predicted AQI based on your input.")

    # ---- Build the same matplotlib figure as the original script ----
    all_years = np.arange(yearly_average["Year"].min(), future_year + 1).reshape(-1, 1)
    regression_line_log = model.predict(all_years)
    regression_line_original = np.exp(regression_line_log)

    fig = plt.figure()
    plt.scatter(x, yearly_average["AQI"], s=100, label="Actual AQI", color="green", zorder=3)
    plt.plot(all_years, regression_line_original, linewidth=2, label="Regression line")
    plt.scatter(
        future_year, predicted_AQI, s=150, marker="x", color="orange",
        label=f"{future_year} Prediction:{predicted_AQI:.2f}", zorder=4
    )
    plt.xlabel("Year")
    plt.ylabel("AQI")
    plt.title(f"AQIs for {city} over the years(2015(or 2018)-2020 plus {future_year} prediction)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # ---- Assemble the response: the printed table + printed messages, as HTML ----
    table_html = yearly_average.to_html(index=False, border=0)
    message_html = table_html + "".join(f"<p>{line}</p>" for line in text_lines)

    return jsonify({
        "message": message_html,
        "image": image_base64
    })


if __name__ == "__main__":
    app.run(debug=True)
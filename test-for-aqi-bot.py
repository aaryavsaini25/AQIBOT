from pandas.io.formats.style_render import Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data="https://raw.githubusercontent.com/adityarc19/aqi-india/refs/heads/main/city_day.csv"
data=pd.read_csv(data)
city=input("For what city in India should I give the AQI data?(Please spell the city correctly with the correct punctuation and capital letters)")
data=data[data['City']==city].copy()
data["Date"]=pd.to_datetime(data["Date"])
data["Year"]=data["Date"].dt.year
data=data.dropna(subset=["AQI","AQI_Bucket"])
data=data.drop(columns=["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"])
yearly_average=(
    data.groupby("Year")["AQI"].mean().reset_index()
)
print(yearly_average)
x=yearly_average[["Year"]]
y_log = np.log(yearly_average["AQI"])
model=LinearRegression()
model.fit(x,y_log)
future_year=int(input("Enter a year to predict the data for it:"))
future_data=pd.DataFrame({"Year":[future_year]})
predicted_log_AQI = model.predict(future_data)[0]
predicted_AQI = np.exp(predicted_log_AQI)
limit=2031
drop=2010
if future_year>limit or future_year<drop:
  print(f"Please do not enter any year that exceeds {limit} or drops below {drop}.")
else:
  print(f"The predicted AQI for {city} for {future_year} is {predicted_AQI:.2f}.")
print(f"Here is a graph for {city}'s historic AQI data and predicted AQI based on your input.")
all_years = np.arange(yearly_average["Year"].min(), future_year + 1).reshape(
            -1, 1
        )
regression_line_log = model.predict(all_years)
regression_line_original = np.exp(
            regression_line_log
        )
plt.scatter(
    x,
    yearly_average["AQI"],
    s=100,
    label="Actual AQI",
    color="green",
    zorder=3
)
plt.plot(
    all_years,
    regression_line_original,
    linewidth=2,
    label="Regression line"
)
plt.scatter(
    future_year,
    predicted_AQI,
    s=150,
    marker="x",
    color="orange",
    label=f"{future_year} Prediction:{predicted_AQI:.2f}",
    zorder=4
)
plt.xlabel("Year")
plt.ylabel("AQI")
plt.title(f"AQIs for {city} over the years(2015(or 2018)-2020 plus {future_year} prediction)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
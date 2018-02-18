from analyze import get_report
import pandas as pd
import copy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    report = get_report().dropna()
    y = pd.DataFrame(report, columns=["pm2.5"])
    # x = copy.deepcopy(report)
    # del x["pm2.5"], x["datetime"], x["cbwd"]
    x = pd.DataFrame(report, columns=["TEMP", "DEWP","month","day","hour","DEWP","TEMP","PRES"])
    # x = pd.DataFrame(report, columns=["TEMP", "DEWP"])
    # x = pd.DataFrame(report, columns=["TEMP"])

    lm = linear_model.LinearRegression()
    model = lm.fit(x, y)
    predictions = lm.predict(x)
    print(mean_squared_error(y,predictions))

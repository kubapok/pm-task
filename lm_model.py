from analyze import get_report
import pandas as pd
import copy
from sklearn import linear_model


if __name__ == "__main__":
    report = get_report()
    pm = pd.DataFrame(report['pm2.5'], columns=["pm2.5"])
    learn = copy.deepcopy(report)
    del learn["pm2.5"], learn["datetime"], learn["cbwd"]
    lm = linear_model.LinearRegression()
    model = lm.fit(learn,pm)

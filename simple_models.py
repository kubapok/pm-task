import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsIC, SGDRegressor
from analyze import get_report_cbwd_encoded
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    report = pd.DataFrame(get_report_cbwd_encoded().dropna())

    target = pd.DataFrame(report, columns=["pm2.5"])
    data = pd.DataFrame(report, columns=[
        'month', 'hour', 'DEWP', 'TEMP', 'PRES',
        'Iws', 'Is', 'Ir', "is_cbwd_NW", "is_cbwd_cv", "is_cbwd_SE", "is_cbwd_NE"])

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.1, random_state=0)

    # naive model (always mean)
    m = float(y_train.mean())
    print('maive MSE: ', mean_squared_error(
        [m for _ in range(len(y_test))], y_test))
    print()

    # linear model
    lm = sm.OLS(y_train, X_train).fit()
    lm.summary()
    print('lm MSE: ', mean_squared_error(lm.predict(X_test), y_test))
    print('lm AIC: ', lm.aic)

    # AIC
    print("AIC")
    aic = LassoLarsIC(criterion='aic')
    aic.fit(X_train, y_train)

    predictions = aic.predict(X_test)
    print(mean_squared_error(y_test, predictions))
    print(aic.coef_)

    # SGD
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    sgd = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)
    sgd = sgd.fit(X_train_scaled, y_train)
    predictions = sgd.predict(scaler.transform(X_test))
    print(mean_squared_error(y_test, predictions))

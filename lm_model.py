from analyze import get_report
import pandas as pd
import copy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


if __name__ == "__main__":
    # lm = linear_model.LinearRegression()
    # model = lm.fit(x, y)
    # predictions = lm.predict(x)
    # print(mean_squared_error(y,predictions))

    report = pd.DataFrame(get_report().dropna(), columns=[
        'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES',
        'cbwd', 'Iws', 'Is', 'Ir'])

    target = pd.DataFrame(get_report().dropna(), columns=["pm2.5"])
    data = pd.DataFrame(report, columns=[
        'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES',
        'Iws', 'Is', 'Ir'])

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.1, random_state=0)
    clf = linear_model.LinearRegression().fit(X_train, y_train)
    # print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X_train, y_train)
    alpha_aic_ = model_aic.alpha_
    print(model_aic.score(X_test, y_test))

    print('PRED')
    predictions = model_aic.predict(X_test)
    print(mean_squared_error(y_test, predictions))

    # svr_rbf = SVR(kernel='rbf')
    # svr = svr_rbf.fit(X_train, y_train)
    # svr.score(X_test, y_test)



    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor

    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    seed = 0
    import numpy
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    numpy.random.seed(seed)
    estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)

    estimator.fit(X_train.values, y_train.values)
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


    # model = lm.fit(x, y)
    # predictions = lm.predict(x)
    # print(mean_squared_error(y, predictions))

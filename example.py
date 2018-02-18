import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from analyze import get_report_cbwd_encoded
import sklearn
from sklearn.metrics import mean_squared_error

report = get_report_cbwd_encoded().dropna()
reportX = pandas.DataFrame(report, columns=[
                           'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
                           'is_cbwd_cv', 'is_cbwd_NW', 'is_cbwd_SE', 'is_cbwd_NE'])
report = pandas.DataFrame(report, columns=['pm2.5'])

scaler = sklearn.preprocessing.MinMaxScaler()
reportX = pandas.DataFrame(scaler.fit_transform(
    reportX), columns=reportX.columns)

datasetX = reportX.values
dataset = report.values
X = datasetX[:, 0:13]
Y = dataset[:, 0]

def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model,
                           nb_epoch=1000, batch_size=4, verbose=0)

estimator.fit(X, Y)
predictions = estimator.predict(X)
print(mean_squared_error(Y, predictions))

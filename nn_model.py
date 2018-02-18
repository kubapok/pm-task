import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from analyze import get_report_cbwd_encoded
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

report = get_report_cbwd_encoded().dropna()
data = pandas.DataFrame(report, columns=[
                           'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
                           'is_cbwd_cv', 'is_cbwd_NW', 'is_cbwd_SE', 'is_cbwd_NE'])
target = pandas.DataFrame(report, columns=['pm2.5'])
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.1, random_state=0)

scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(X_train)
X_trained_scaled = pandas.DataFrame(scaler.transform(X_train))

X_trained_scaled = X_trained_scaled.values
y_train = y_train.values
X = X_trained_scaled[:, 0:13]
Y = y_train[:, 0]

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


X_test_scaled = pandas.DataFrame(scaler.transform(X_test))
X_test_scaled = X_test_scaled.values
y_test = y_test.values
X = X_test_scaled[:, 0:13]
Y = y_test[:, 0]

predictions = estimator.predict(X)
print(mean_squared_error(Y, predictions))

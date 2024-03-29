import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from analyze import get_report_cbwd_encoded

report = get_report_cbwd_encoded().dropna()
data = pd.DataFrame(report, columns=[
    'month', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
    'is_cbwd_cv', 'is_cbwd_NW', 'is_cbwd_SE', 'is_cbwd_NE'])
target = pd.DataFrame(report, columns=['pm2.5'])
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.1, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train)).values[:, 0:X_train.shape[1]]
y_train = y_train.values[:, 0]


def nn_model():
    model = Sequential()
    model.add(Dense(20, input_dim=X_train.shape[1],
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, input_dim=40,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(500, input_dim=500,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(500, input_dim=500,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, input_dim=40,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, input_dim=20,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=nn_model,
                           nb_epoch=10, batch_size=1, verbose=1)



# slow, but more accurate
# from sklearn.model_selection import cross_val_score
# results = cross_val_score(estimator, X, y, cv=10)
# print(results.mean(), results.std())
# import sys; sys.exit(0)

estimator.fit(X_train_scaled, y_train)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test)).values[:, 0:X_train.shape[1]]
y_test = y_test.values[:, 0]

predictions = estimator.predict(X_test_scaled)
print(mean_squared_error(y_test, predictions))

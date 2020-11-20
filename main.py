from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

X_train = pd.read_csv('Training.csv')
X_test = pd.read_csv('Testing.csv')

y_train = X_train.pop('OUTPUT')
y_test = X_test.pop('OUTPUT')

feature_columns = []
for feature_name in ['avg_wind_speed', 'sol_rad', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'avg_soiltemp_4in_sod']:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))

def make_input_fn(data_df, label_df, num_epochs=50, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(X_train, y_train)
test_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(test_input_fn)

print('Model stats:')
print(result)

def make_input_fn(features, batch_size=32):
  def input_function():
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
  return input_function

# Given an index from the testing set we can find the predicted value
# In this case I am printing the predicted value of the first index in the testing set
print(list(linear_est.predict(make_input_fn(X_test)))[0]['predictions'][0])

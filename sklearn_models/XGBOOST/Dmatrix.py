data = np.array([
  [1.2, 3.3, 1.4],
  [5.1, 2.2, 6.6]])

import xgboost as xgb
dmat1 = xgb.DMatrix(data)

labels = np.array([0, 1])
dmat2 = xgb.DMatrix(data, label=labels)
#adding a booster
# predefined data and labels
print('Data shape: {}'.format(data.shape))
print('Labels shape: {}'.format(labels.shape))
dtrain = xgb.DMatrix(data, label=labels)

# training parameters
params = {
  'max_depth': 0,
  'objective': 'binary:logistic',
  'eval_metric':'logloss'
}
print('Start training')
bst = xgb.train(params, dtrain)  # booster
print('Finish training')


#using a booster
# predefined evaluation data and labels
print('Data shape: {}'.format(eval_data.shape))
print('Labels shape: {}'.format(eval_labels.shape))
deval = xgb.DMatrix(eval_data, label=eval_labels)

# Trained bst from previous code
print(bst.eval(deval))  # evaluation

# new_data contains 2 new data observations
dpred = xgb.DMatrix(new_data)
# predictions represents probabilities
predictions = bst.predict(dpred)
print('{}\n'.format(predictions))


#cv
# predefined data and labels
dtrain = xgb.DMatrix(data, label=labels)
params = {
  'max_depth': 2,
  'lambda': 1.5,
  'objective':'binary:logistic',
  'eval_metric':'logloss'

}
cv_results = xgb.cv(params, dtrain)
print('CV Results:\n{}'.format(cv_results))

#saving
# predefined data and labels
dtrain = xgb.DMatrix(data, label=labels)
params = {
  'max_depth': 3,
  'objective':'binary:logistic',
  'eval_metric':'logloss'
}
bst = xgb.train(params, dtrain)

# 2 new data observations
dpred = xgb.DMatrix(new_data)
print('Probabilities:\n{}'.format(
  repr(bst.predict(dpred))))

bst.save_model('model.bin')

#loading
# Load saved Booster
new_bst = xgb.Booster()
new_bst.load_model('model.bin')

# Same dpred from before
print('Probabilities:\n{}'.format(
  repr(new_bst.predict(dpred))))

#XGBClassifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# predefined data and labels
model.fit(data, labels)

# new_data contains 2 new data observations
predictions = model.predict(new_data)
print('Predictions:\n{}'.format(repr(predictions)))

model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', use_label_encoder=False)
# predefined data and labels (multiclass dataset)
model.fit(data, labels)

# new_data contains 2 new data observations
predictions = model.predict(new_data)
print('Predictions:\n{}'.format(repr(predictions)))
model = xgb.XGBRegressor(max_depth=2)
# predefined data and labels (for regression)
model.fit(data, labels)

# new_data contains 2 new data observations
predictions = model.predict(new_data)
print('Predictions:\n{}'.format(repr(predictions)))

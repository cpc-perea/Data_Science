# Databricks notebook source
# DBTITLE 1,Step 1: Import Packages
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

# COMMAND ----------

# DBTITLE 1,Step 2: Access Training Data
df = spark.read.format("csv").option("header", "true").load('/mnt/edp-sandbox/dpo_dr_datasets/7-1/dpo_training_lab_anchored_30min_offset_latest.csv').toPandas()

# COMMAND ----------

# DBTITLE 1,Step 3: Access SpecWeb data
spec_web = spark.read.parquet('/mnt/refined-zone/SpecWeb/specweb.parquet')
spec_web = spec_web.toPandas()

# COMMAND ----------

# DBTITLE 1,Step 4: Set index for training data
df = df.set_index('SampleTime').sort_index()
df.shape

# COMMAND ----------

# DBTITLE 1,Step 5: Filter training data on various conditions
#CONFIRM WITH BRENT
#df = df[df["Gear Bearing Stable Temperatures"] == "1.0"]
df = df[df["Extruder Status"]=="RUN"]

# COMMAND ----------

# DBTITLE 1,Step 6: Remove unnecessary data
#drop variables that are specific to Seeq
df = df.drop(['_c0', 'Averaging Windows', 'Averaging Windows - Value', 'Gear Bearing Stable Temperatures - _SeeqInternal_startClipped'], axis=1)

# COMMAND ----------

#drop variables that are unneeded based on feedback 
#to be modified as needed based on future feedback
df = df.drop(['Extruder Status',
              'Motor KW / Gear Pump Amps Ratio',
              'Gear Pump Bearing 1 Temp',
              'Gear Pump Bearing 2 Temp',
              'Gear Pump Bearing 3 Temp',
              'Gear Pump Bearing 4 Temp',
              'Gear Bearing Stable Temperatures', 
              'Reactor Production Rate'], axis=1)

# COMMAND ----------

#drop variables that are unneeded based on data quality/completeness
df = df.drop(['Extruder Drive Motor (low speed)', 'Zone 0 Temp'], axis=1)

# COMMAND ----------

#drop rows with missing Resin, MI, HLMI data
#df = df.dropna(subset=['Resin (Extruder)', 'Melt Index (pellets)', 'HLMI (pellets)'], how='all')

# COMMAND ----------

# drop all nulls except in melt index, hlmi columns
df = df.dropna(subset=df.columns.difference(['Melt Index (pellets)', 'HLMI (pellets)']))

# COMMAND ----------

# DBTITLE 1,Step 6.5: Reclassify resins
#reclassify resins based on CPChem's feedback
df= df.replace({'Resin (Extruder)': {'HHM 5202-02BN': 'HHM 5202BN'}})

# COMMAND ----------

# DBTITLE 1,Step 7: Apply numeric datatype
cols = [i for i in df.columns if i not in ['Resin (Extruder)']]
for col in cols:
  df[col] = pd.to_numeric(df[col])

# COMMAND ----------

# DBTITLE 1,Step 8: Remove data based on thresholds and control limits
#drop data based on feedback regarding control limits from CPChem (handle outliers)
cols = ['Zone 1 Temp',
        'Zone 2 Temp',
        'Zone 3 Temp',
        'Zone 4 Temp',
        'Zone 5 Temp',
        'Zone 6 Temp',
        'Zone 7 Temp',
        'Zone 8 Temp',
        'Zone 9 Temp',
        'Zone 10 Temp']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 5 * IQR)) |(df[cols] > (Q3 + 5 * IQR))).any(axis=1)]

# COMMAND ----------

#drop data above a threshold provided from CPChem (handle anomalies)
zones_list = df[cols].columns[(df[cols] > 1000).any()].to_list()
df = df.drop(columns=zones_list, axis=1)

# COMMAND ----------

# DBTITLE 1,Step 9: Integrate SpecWeb data
resin_list = df['Resin (Extruder)'].unique().tolist()
spec_web['productNumber'] = spec_web['productNumber'].str.strip()
spec_web['productNumber'] = spec_web['productNumber'].replace({'HXM 50100-01': 'HXM 50100-1'})
spec_web[spec_web['productNumber'].isin(resin_list)]

# COMMAND ----------

spec_web = spec_web[(spec_web['minimum'] != '---') | (spec_web['maximum'] != '---')]

# COMMAND ----------

df = df.merge(spec_web, how='left', left_on='Resin (Extruder)', right_on='productNumber').set_index(df.index).drop(['productNumber', 'revisionDate', 'approvedProductionSite','sortType', 'type', 'method', 'units', 'target', 'sortOrder'], axis=1).rename({'minimum': 'lcl', 'maximum': 'ucl'}, axis=1)

# COMMAND ----------

# DBTITLE 1,Step 10: Process MI and HLMI assignments
HLMI_resins = df[df['item_name']=='HLMI']['Resin (Extruder)'].unique().tolist()
df.loc[df['Resin (Extruder)'].isin(HLMI_resins), 'Melt Index (pellets)'] = np.nan
df.loc[~df['Resin (Extruder)'].isin(HLMI_resins), 'HLMI (pellets)'] = np.nan

# COMMAND ----------

#drop rows when both mi and hlmi values are missing
df = df.dropna(subset=['Melt Index (pellets)', 'HLMI (pellets)'], how='all')

# COMMAND ----------

df['Resin_Type'] = np.where(df['HLMI (pellets)'].isnull(), 'MI', 'HLMI')

# COMMAND ----------

# DBTITLE 1,Step 11: Filter outliers in training data
cols = [i for i in df.columns if i not in ['Resin (Extruder)', 'item_name', 'Resin_Type']]
for col in cols:
    df[col] = pd.to_numeric(df[col])
    
#filter outliers 
df = df[((df['Melt Index (pellets)'] >= (df['lcl'] - (df['lcl']*0.5))) & (df['Melt Index (pellets)'] <= (df['ucl'] + (df['ucl']*0.5)))) | ((df['HLMI (pellets)'] >= (df['lcl'] - (df['lcl']*0.5))) & (df['HLMI (pellets)'] <= (df['ucl'] + (df['ucl']*0.5))))]

# COMMAND ----------

# DBTITLE 1,Step 12: Remove data based on proportion represented in training data
#drop resins that make up less than 1% of training dataset
resin_counts = df['Resin (Extruder)'].value_counts(normalize=True)
to_remove = resin_counts[resin_counts <= 0.01].index
df = df[~df['Resin (Extruder)'].isin(to_remove)]

# COMMAND ----------

# DBTITLE 1,Step 13: Replace negative fluff feed rates by 0
df['Extruder Fluff Feed Rate'] = np.where(df['Extruder Fluff Feed Rate'] < 0, 0, df['Extruder Fluff Feed Rate'])

# COMMAND ----------

# DBTITLE 1,Step 14: Create new dummy features for resin types
df = pd.get_dummies(df, columns=['Resin (Extruder)'])

# COMMAND ----------

# DBTITLE 1,Step 15: Remove unnecessary SpecWeb data
df.drop(['item_name', 'lcl', 'ucl'], axis=1, inplace=True)
df.shape

# COMMAND ----------

# DBTITLE 1,Step 16: Map MI and HLMI assignments
df['Resin_Type'] = df['Resin_Type'].map({'HLMI': 1, 'MI': 0})

# COMMAND ----------

# DBTITLE 1,Step 17: Split data for model training
X =  df.drop(['Melt Index (pellets)', 'HLMI (pellets)'], axis=1)
y = df[['Melt Index (pellets)', 'HLMI (pellets)']]

# COMMAND ----------

X_train_size = int(len(X) * 0.9)
X_train, X_test = X[0:X_train_size], X[X_train_size:]

y_train_size = int(len(y) * 0.9)
y_train, y_test = y[0:y_train_size], y[y_train_size:]

# COMMAND ----------

# DBTITLE 1,Step 18: Initialize and fit scalers
scaler_mi = RobustScaler()
scaler_hlmi = RobustScaler()

# COMMAND ----------

y_train['Melt Index (pellets)'] = scaler_mi.fit_transform(y_train[['Melt Index (pellets)']])
y_train['HLMI (pellets)'] = scaler_hlmi.fit_transform(y_train[['HLMI (pellets)']])

y_test['Melt Index (pellets)'] = scaler_mi.transform(y_test[['Melt Index (pellets)']])
y_test['HLMI (pellets)'] = scaler_hlmi.transform(y_test[['HLMI (pellets)']])

# COMMAND ----------

# DBTITLE 1,Step 19: Process training and testing data
y_train['HLMI/MI'] = y_train['Melt Index (pellets)'].fillna(y_train['HLMI (pellets)'])
y_train.drop(['Melt Index (pellets)', 'HLMI (pellets)'], axis=1, inplace=True)

y_test['HLMI/MI'] = y_test['Melt Index (pellets)'].fillna(y_test['HLMI (pellets)'])
y_test.drop(['Melt Index (pellets)', 'HLMI (pellets)'], axis=1, inplace=True)

# COMMAND ----------

# DBTITLE 1,Step 20: Fit model with optimal parameters
xgb = XGBRegressor(random_state= 123)

params = [{'learning_rate': [0.01, 0.1], 
           'max_depth': [3, 5, 7, 10], 
           'min_child_weight': [1, 3, 5, 7], 
           'subsample': [0.5, 0.7, 1], 
           'colsample_bytree': [0.5, 0.7, 1], 
           'n_estimators' : [100, 200, 500, 700]}]

xgb_cv = RandomizedSearchCV(xgb, param_distributions=params, cv = TimeSeriesSplit(n_splits=10), scoring='neg_mean_absolute_error')
xgb_cv.fit(X_train, y_train)
y_pred_xgb_hyp_test = xgb_cv.predict(X_test)

# COMMAND ----------

xgb_cv.best_params_

# COMMAND ----------

# DBTITLE 1,Step 21: Helper functions for inversing scales and assessing performance
def inverse_scale_actuals(df):
  if df['Resin_Type']==1.0:
    val = df['Scaled Actuals']
    transformed_val = scaler_hlmi.inverse_transform([[val]])
    return transformed_val[0][0]
  elif df['Resin_Type']==0.0:
    val = df['Scaled Actuals']
    transformed_val = scaler_mi.inverse_transform([[val]])
    return transformed_val[0][0]

# COMMAND ----------

def inverse_scale_predictions(df):
  if df['Resin_Type']==1.0:
    val = df['Scaled Predictions']
    transformed_val = scaler_hlmi.inverse_transform([[val]])
    return transformed_val[0][0]
  elif df['Resin_Type']==0.0:
    val = df['Scaled Predictions']
    transformed_val = scaler_mi.inverse_transform([[val]])
    return transformed_val[0][0]

# COMMAND ----------

def show_metrics(df):
  print('Mean Absolute Error:', metrics.mean_absolute_error(df['Inverse Actuals'], df['Inverse Predictions']). round(4))
  print('Mean Absolute Percentage Error:', np.mean((np.abs((df['Inverse Actuals'] - df['Inverse Predictions'])/df['Inverse Actuals']))*100).round(4), '%')
  print('Model Accuracy:', ((1-(np.mean(np.abs((df['Inverse Actuals'] - df['Inverse Predictions'])/df['Inverse Actuals']))))*100).round(4), '%')
  print('Mean Squared Error:', metrics.mean_squared_error(df['Inverse Actuals'], df['Inverse Predictions']). round(4))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Inverse Actuals'], df['Inverse Predictions'])). round(4))
  print('Coefficient of Determination:', metrics.r2_score(df['Inverse Actuals'], df['Inverse Predictions']). round(4))

# COMMAND ----------

# DBTITLE 1,Step 22: Assess model performance
y_pred_xgb_hyp_test = pd.merge(pd.merge(X_test['Resin_Type'], pd.DataFrame(y_pred_xgb_hyp_test).set_index(X_test.index), on='SampleTime', how='inner'), y_test, on='SampleTime')
y_pred_xgb_hyp_test.rename(columns = {0:"Scaled Predictions", 'HLMI/MI': 'Scaled Actuals'}, inplace="True")

y_pred_xgb_hyp_test['Inverse Predictions'] = y_pred_xgb_hyp_test.apply(inverse_scale_predictions, axis=1)
y_pred_xgb_hyp_test['Inverse Actuals'] = y_pred_xgb_hyp_test.apply(inverse_scale_actuals, axis=1)

show_metrics(y_pred_xgb_hyp_test)

# COMMAND ----------

y_pred_xgb_hyp_test_high = y_pred_xgb_hyp_test[y_pred_xgb_hyp_test['Resin_Type']==1.0]
y_pred_xgb_hyp_test_low = y_pred_xgb_hyp_test[y_pred_xgb_hyp_test['Resin_Type']==0.0]

# COMMAND ----------

show_metrics(y_pred_xgb_hyp_test_high)

# COMMAND ----------

show_metrics(y_pred_xgb_hyp_test_low)

# COMMAND ----------

# DBTITLE 1,Step 23: Save Predictive Model (XGBoost)
import pickle
bundle_filepath = '/dbfs/mnt/edp-sandbox/dpo_dev/data_science/7-1/DigitalRheometer_models_7-1_v3.pkl'

with open(bundle_filepath, 'wb') as bundle:
  pickle.dump({
    'scaler_HLMI': scaler_hlmi,
    'scaler_MI': scaler_mi,
    'model': xgb_cv
  }, bundle)

# COMMAND ----------


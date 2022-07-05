# Databricks notebook source
# MAGIC %md 
# MAGIC ### initialization

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy as cp

p_y_cols = [ 'Melt Index (pellets)', 'HLMI (pellets)']
p_resin_col = 'Resin (Extruder)'
p_status_col = 'Extruder Status'
p_status_on_val = 'RUN'
p_date_col = 'SampleTime'
p_data_file = '/mnt/edp-sandbox/dpo_dr_datasets/consolidated/PPC/signal_data.parquet'
p_extruder = '7-1'
p_site = 'PPC'

p_exclude_cols = ['Averaging Windows', 'Extruder Drive Motor (low speed)', 'Polymer Melt Temp at Die', 'Zone 0 Temp', 'Zone 9 Temp']

# COMMAND ----------

# MAGIC %md
# MAGIC ### read raw data

# COMMAND ----------

df_raw = spark.read.parquet(p_data_file).toPandas()
df_start = df_raw if p_extruder is None else df_raw.loc[df_raw.Extruder == p_extruder].drop(columns = 'Extruder').reset_index(drop = True)
df_start

# COMMAND ----------

# MAGIC %md 
# MAGIC ### preprocessing classes

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

# COMMAND ----------

# MAGIC %md
# MAGIC #### class to calculate run

# COMMAND ----------

class MLRRunTransitionCalculator(BaseEstimator, TransformerMixin): 
  """
  This will create a column in the data called 'Run' and 'Transition for analytical purpose only. This is to group together points in time to analysis.
  
  Initialization parameters: 
  MLRRunTransitionCalculator(resin_col, status_col, status_on_val, extruder_col = 'Extruder', date_col = 'SampleTime')
  
  resin_col : the column that represent the resin being produced on the extruder.
  status_col : the column that represents the extruder run status. 
  status_on_val : the value that represent when the status is on for the extruder. 
  extruder_col : the column that contains the extruder name. Default = 'Extruder'
  date_col : the column that represent the timestamp of the sample. Default = 'SampleTime',
  calc_run_col :
  """
  def __init__(self, resin_col, status_col, status_on_val, extruder_col = 'Extruder', date_col = 'SampleTime'):
    self.date_col = date_col
    self.resin_col = resin_col
    self.status_col = status_col
    self.status_on_val = status_on_val
    self.extruder_col = extruder_col
    return
  
  def fit(self, X, y = None):
    return self
  
  def transform(self, X, y = None): 
    #for use in later parts of the transformation. 
    X = cp.deepcopy(X)
    extruder_present = self.extruder_col in X.columns
    if (not extruder_present):
      X[self.extruder_col] = 'TMP'
    
    # sort data by extruder then SampleDate. If not extruder, then just sample date
    sort_cols = [self.extruder_col, self.date_col]
    X = X.sort_values(sort_cols).reset_index(drop = True)
    
    # drop rows with no status value
    print(f'Dropping {X[self.status_col].isna().sum()} samples due to no status column value.')
    X = X[~X[self.status_col].isna()].reset_index(drop = True)
    
    # Calculate the shift columns
    shift_columns = ['R-1', 'S-1', 'D-1']
    start_cols = [self.date_col, self.resin_col, self.extruder_col, self.date_col] 
    X_lag = cp.deepcopy(X[start_cols])
    X_lag[shift_columns] = X.groupby(self.extruder_col)[[self.resin_col, self.status_col, self.date_col]].shift(1)
    X_lag[['R+1', 'D+1']] = X.groupby(self.extruder_col)[[self.resin_col, self.date_col]].shift(-1)
    
    # Calculate the runs. 
    row_select = ( 
      X[self.status_col].ne(X_lag['S-1']) 
      & X[self.status_col].eq(self.status_on_val)
      & ( ( X[self.date_col] - X_lag['D-1'] ).dt.total_seconds().le(12*60*60) | X_lag['D-1'].isna() )
    )
    X_lag['run_changes'] = 0
    X_lag.loc[row_select, 'run_changes'] = 1
    X['Run'] = X_lag.groupby([self.extruder_col])['run_changes'].cumsum().map(lambda r: f"Run {r:03d}")
    
    #calculate the transition samples (resin switches)
    row_select = (
      ( X[self.resin_col].ne(X_lag['R-1']) | X[self.resin_col].ne(X_lag['R+1']) )
      & ( ( X[self.date_col] - X_lag['D-1'] ).dt.total_seconds().le(12*60*60) | X_lag['D-1'].isna() )
    )
    X['Transition'] = False
    X.loc[row_select, ['Transition']] = True
    
    # Drop the temp extrude column if no extruder column was present in the original.
    if not extruder_present:
      X.drop(columns = 'Extruder', inplace = True)
    
    # drop samples that are not in the run = on status. return the dataset. 
    print(f'Dropping {X[self.status_col].ne(self.status_on_val).sum()} samples because run status is not \'{self.status_on_val}\'.')    
    return X[X[self.status_col] == self.status_on_val].reset_index(drop = True)
  
  def fit_transform(self, X, y = None):
    print(f'received y [runcalc]: {y}')
    self.fit(X, y)
    return self.transform(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### class to calculate sample dTs and dS

# COMMAND ----------

class MLRCalculate_dS_dT(BaseEstimator, TransformerMixin):
  """
  This class will calculate the differentials and lags for the specified columns. It will also put in the column 'dT' the number of hours from the prior sample. 
  
  MLRCalculate_dS_dT(resin_col, lag_cols, diff_cols = None, shifts = list(range(1, 4)), lag_dT_cutoff_hrs = 12, date_col = 'SampleTime', extruder_col = 'Extruder')
  
  Initialization Parameters: 
  resin_col           : The column that specifies the resin currently in production. 
  date_col            : The column that specifies the date column. Defaulted to 'SampleTime'
  lag_cols            : The columns to get lags for.
  diff_cols           : The columns to get differentials for. None will cause it to get all floating point column differentials. 
  shifts              : A list of shifts to execute. More that one item in this list will generate more than one sample for the same current record with a different dT. 
  lag_dT_cutoff_hrs   : The filter for # hrs over which to exclude lag dT samples.
  extruder_col        : The column that represents the extruder.
  y_col               : the y column, will not diff. 
  """
  def __init__(self, 
               resin_col, 
               lag_cols, 
               diff_cols = None,
               shifts = list(range(1, 4)), 
               lag_dT_cutoff_hrs = 12, 
               date_col = 'SampleTime', 
               extruder_col = 'Extruder',
               y_col = 'y'):
    self.resin_col = resin_col
    self.date_col = date_col
    self.lag_cols = lag_cols if type(lag_cols) == list else list(lag_cols)
    self.diff_cols = diff_cols
    self.shifts = shifts
    self.lag_dT_cutoff_hrs = lag_dT_cutoff_hrs
    self.extruder_col = extruder_col
    self.y_col = y_col
    return
  
  def fit(self, X, y = None):
    # if no columns specified get the differentials for all floating point columns
    if self.diff_cols is None:
      self.diff_cols_ = X.select_dtypes(['float', 'float32', 'float64']).columns.to_list()
    else:
      self.diff_cols_ = list(self.diff_cols)
    
    # remove the y_col from the diff_cols_
    if self.y_col in self.diff_cols_:
      self.diff_cols_.remove(self.y_col)
    return self
  
  def transform(self, X, y = None):
    # copy the dataset then ensure there is an extruder field (used for group bys.)
    X = cp.deepcopy(X)
    extruder_present = self.extruder_col in X.columns
    if (not extruder_present):
      X[self.extruder_col] = 'TMP'
    
    print(f'Dropping {X.shape[0] - X[~X[self.lag_cols + self.diff_cols_].isna().any(axis = 1)].shape[0]} samples due to NaN value in the record. Will do not imputation.')
    X = X[~X[self.lag_cols + self.diff_cols_].isna().any(axis = 1)].reset_index(drop = True)
    
    # Get the diffs for each specified shifts. 
    dfs_diffed = []
    for num_shifts in self.shifts:
      Xt = self.calc_time_diff(X, num_shifts)
      Xt = self.calc_lags(Xt, num_shifts)
      Xt = self.calc_diffs(Xt, num_shifts)
      dfs_diffed.append(Xt)
    
    # concatenate the dataframes together that were generated above and sort by extruder date and dT. 
    df_ret = pd.concat(dfs_diffed).sort_values([self.extruder_col, self.date_col, 'dT'])
    
    # filter out the samples with dT > lag_dT_cutoff_hrs
    print(f'Dropping {df_ret.dT.gt(self.lag_dT_cutoff_hrs).sum()} samples due to lag being more than {self.lag_dT_cutoff_hrs} hours before {self.date_col}.')
    df_ret = df_ret.loc[df_ret.dT.between(0.1, self.lag_dT_cutoff_hrs)]
    
    # remove extruder column if the dataset did not come with one. 
    if not extruder_present: 
      df_ret.drop(columns = 'Extruder', inplace = True)
      
    # return the transformed dataset. 
    self.named_features_out_ = df_ret.columns.to_list()
    return df_ret.reset_index(drop = True)
   
  # calculate the dT from the lag (num_shifts). dT is deltatime. 
  def calc_time_diff(self, df, num_shifts):
    df = cp.deepcopy(df)
    df.loc[:, 'dT'] = df.groupby(self.extruder_col)[self.date_col].shift(num_shifts)
    df.loc[:, 'dT'] = df.loc[:, self.date_col] - df.loc[:, 'dT']
    df.loc[:, 'dT'] = df.loc[:, 'dT'].dt.total_seconds() / (60*60)
    return df
  
  # calculate the lag columns. Prevent bleed over from extruder to extruder. 
  def calc_lags(self, df, num_shifts):
    for col in self.lag_cols: 
      df.loc[:, f'{col}_prev'] = df.groupby(self.extruder_col)[col].shift(num_shifts)
    return df
  
  # calculate the differentials. 
  def calc_diffs(self, df, num_shifts):
    df = cp.deepcopy(df)
    col_names = {}
    for col in self.diff_cols_:
      col_names[col] = f'd({col})'
    for col in self.diff_cols_:
      df.loc[:, col_names[col]] = df.groupby(self.extruder_col)[col].diff(num_shifts)

    return df
  
  def fit_transform(self, X, y = None):
    print(f'received y [dSdT]: {y}')
    self.fit(X, y)
    return self.transform(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### class to consolidate mi and hlmi

# COMMAND ----------

class MLRMiHlmiCombiner(BaseEstimator, TransformerMixin):
  """
  MLRMiHlmiCombiner(resin_col, y_cols, y_col = 'y', test_type_col = 'Test Type', 
                    test_types = ['MI', 'HLMI'], test_type_det = 'specs', resin_map, spec_site, 
                    spec_table = { 'type':'file', 'location':'/mnt/refined-zone/SpecWeb/specweb.parquet', 'format':'parquet' }, 
                    offspec_col = False, 
                    convert_hlmi_to_mi = True)
  
  This class is used to transform the data in a first step to combine the MI/HLMI columns based on either 
  a voting system or by the test prescribed by the product specs from SpecWeb Tool.

  This class does a "fit" to remember MI test type is used for each column for transformations. 

  There is no inverse transforms from this class. 

  Constructor Parameters: 
    resin_col:          This is the column of the dataset has houses the "Resin" name. 
    y_cols:             A list containing the MI and HLMI columns. MI is expected first, then HLMI. 
    y_col:              (default: "y") The name of the combined MI/HLMI column after the transformation.
    test_type_col:      (default: "Test Type") The name of the column that receives the categorical variable of the MI test type.
    test_types:         (default: ['MI', 'HLMI'] the names of the test type that will the values of the test_type_col of the trasformed dataset)
    test_type_det:      (default: "specs") Determines which Y col to used (HI or HLMI) by either 1. examining the dataset and using a voting mechanism,
                        or gets the information from SpecWeb. Valid values : "specs", "vote".
    resin_map:          Resin names coming from the DCS/Historians may not match SpecWeb. There is a resin map by default.
                        This is only needed when using "spec" for test_type_det. "vote" will not have this problem, but may miss new resins.
    spec_site:          which site should we get product specifications for? 
    spec_table:         3 to 4 value dictionary [ site, type (file or table), location (file path or location), format (csv, parquet, delta)]
    offspec_col         (default: False) if test_type_det == 'specs', True would generate a categorical offspec column `Offspec` for samples outside control limits.
    convert_hlmi_to_mi: (default: True) convert the HLMI to MI by dividing by 100. 
  """
  def __init__(
    self, 
    resin_col, 
    y_cols, 
    y_col = "y", 
    test_type_col = "Test Type",
    test_types = ['MI', 'HLMI'],
    test_type_det = "specs",
    resin_map = { 
      'HXM 50100-1': 'HXM 50100-01',
      'HHM 5202-02BN': 'HHM 5202BN'
    },
    spec_site = None,
    spec_table = { 'type':'file', 'location':'/mnt/refined-zone/SpecWeb/specweb.parquet', 'format':'parquet' },
    offspec_col = False,
    convert_hlmi_to_mi = True
  ): 
    self.resin_col = resin_col
    self.y_cols = y_cols
    self.y_col = y_col
    self.test_type_col = test_type_col
    self.test_types = test_types 
    self.test_type_det = test_type_det
    self.resin_map = resin_map
    self.spec_table = spec_table
    self.spec_site = spec_site 
    self.spec_maps = { 'Melt Index':'MI', 'HLMI':'HLMI' }
    self.offspec_col = offspec_col
    self.convert_hlmi_to_mi = convert_hlmi_to_mi
    
    if test_type_det == 'specs' and spec_site is None:
      raise(Exception('ERROR: When test_type_det = \'specs\', spec_site must be specified.'))
      
  def fit(self, X, y = None):
    """Fits the transform model and persists which resin uses MI which resin uses HLMI, how to determine is either vote or specs as specified in the constructor."""
    # rename the y_cols to the test_type names
    X = cp.deepcopy(X)
    X = X.rename(columns = dict(zip(self.y_cols, self.test_types)))
    self.test_types_ = X.columns[X.columns.isin(self.test_types)].to_list()

    if self.test_type_det == 'vote': 
      # determine, by vote, which y_column to use by distinct resin_cols values
      votes = X.melt(id_vars = [ self.resin_col ] , value_vars = self.test_types_).pivot_table(index = self.resin_col, columns = 'variable', values = 'value', aggfunc = 'count')
      votes = votes.idxmax(axis = 1)
      self.specs_ = pd.DataFrame(votes.rename(self.test_type_col))
    
    elif self.test_type_det == 'specs': 
      ########################################################################
      # This lookup should come from data. I don't if or where we have this. #
      ########################################################################
      siteLookup = {
        'Cedar Bayou': 'CED',
        'Orange': 'ORA',
        'Pasadena Plastics Complex': 'PPC',
        'Old Ocean': 'SWE'
      }
      specs = spark.read.format(self.spec_table['format']).option('inferSchema', 'true').load(self.spec_table['location']).toPandas()
      specs['site'] = specs.apply(
        lambda r: siteLookup[r['approvedProductionSite']],
        axis = 1
      )
      specs.productNumber = specs.productNumber.str.strip()
      specs[self.test_type_col] = specs['item_name'].map(self.spec_maps)
      specs = specs[~specs[self.test_type_col].isna() & (specs['site'] == self.spec_site)].reset_index()
      specs = specs[(specs.target + specs.minimum + specs.maximum != '---------')]
      specs = specs[['productNumber', 'target', 'minimum', 'maximum', self.test_type_col]].reset_index(drop = True)
      specs['minimum'] = specs['minimum'].map(float)
      specs['maximum'] = specs['maximum'].map(float)
      specs['target']  = specs['target'].map(lambda r: float(r) if r.replace('.', '', 1).isdigit() else np.nan)      
      self.specs_ = specs.set_index('productNumber')
      
    return self   

  def transform(self, X, y = None):
    X = cp.deepcopy(X)
    
    # if the test type determination was via a vote, the map the resin col values.
    if self.test_type_det == 'specs':
      X[self.resin_col] = X[self.resin_col].map(lambda r: r if r not in self.resin_map else self.resin_map[r])
    
    # remove rows with no resin column value
    print(f'Dropping {X[self.resin_col].isna().sum()} samples due to no resin in [{self.resin_col}].')
    X = X.loc[~X[self.resin_col].isna()].reset_index(drop = True)
    
    # rename the y_cols to the test_type names
    X = X.rename(columns = dict(zip(self.y_cols, self.test_types))).reset_index(drop = True)
    
    # initialize the Test Type and y_cal, and offspec columns (* if specified)
    X[self.test_type_col] = ''    

    X[self.y_col] = np.nan
    if (self.offspec_col):
      X['Offspec Status'] = 'Normal'
    
    # loop through the distinct resins in the dataset. 
    for r in X[self.resin_col].unique():
      # check to see if resin has specification
      if (r not in self.specs_.index):
        rcnt = X[X[p_resin_col] == r].shape[0]
        print(f'WARNING: Resin [{r}] does not contain a specification, dropping [{rcnt}] rows.')
        X = X.loc[X[p_resin_col] != r].reset_index(drop = True)
        continue
       
      # set the Test Type column value
      X.loc[X[self.resin_col].eq(r), self.test_type_col] = self.specs_.loc[r][self.test_type_col]
            
      # Consolidate the Y value cols into the y_val col. 
      # check to see if the Test Type Column column exists in the fitting data.
      # i.e. if the resin is a HLMI resin, make sure the HLMI column exists. If not, then print warning.
      if (self.specs_.loc[r][self.test_type_col] not in X.columns):
        print('WARNING: Resin [%s] uses the %s test, but this column does not exist in the data. Removing this resin from the dataset.' %
              (r, self.specs_.loc[r][self.test_type_col]))
        X = X.loc[~X[p_resin_col].eq(r)].reset_index(drop = True)
        continue

      # consolidate the mi and hlmi columns into a single column. 
      X.loc[X[self.resin_col].eq(r), self.y_col] = X.loc[X[self.resin_col].eq(r), self.specs_.loc[r][self.test_type_col]]

      # if there is a requirement for an offspec indicator, add it. 
      if (self.offspec_col):
        Xt = X.loc[X[self.resin_col].eq(r)]
        X.loc[X[self.resin_col].eq(r), 'Offspec Status'] = \
          np.where(Xt[self.y_col] < self.specs_.loc[r]['minimum'], 'Low', Xt['Offspec Status'])
        Xt = X.loc[X[self.resin_col].eq(r)]
        X.loc[X[self.resin_col].eq(r), 'Offspec Status'] = \
          np.where(Xt[self.y_col] > self.specs_.loc[r]['maximum'], 'High', Xt['Offspec Status'])

    # if convert hlmi to mi, do so.
    if self.convert_hlmi_to_mi:
      X.loc[X[self.test_type_col].eq(self.spec_maps['HLMI']), self.y_col] \
        = (X.loc[X[self.test_type_col].eq(self.spec_maps['HLMI']), self.y_col] / 100.0)
          
    # remove training values with no Y value
    print(f'Dropping {(X[self.y_col].isna() | X[self.y_col].eq(0)).sum()} samples due to having no MI or HLMI.')
    X = X.loc[~(X[self.y_col].isna() | X[self.y_col].eq(0))]

    # remove original y_columns
    X = X.drop(self.test_types_, axis = 1, errors = 'ignore')

    return X.reset_index(drop = True)
  
  def fit_transform(self, X, y = None):
    print(f'received y [mi_hlmi]: {y}')
    self.fit(X, y)
    return self.transform(X, y)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### class to select specific columns and dynamically drop NA columns

# COMMAND ----------

class MLRColumnSelector(BaseEstimator, TransformerMixin):
  """
  MLRColumnSelector(self, drop_empty_cols = True, cols_to_drop = None, cols_to_keep = None)
  
  This class can be used to:
  
  1. Drop columns with no data. 
  2. Specify a set of columns to keep.
  3. Specify a set of columns to drop. 
  """
  def __init__(self, drop_empty_cols = True, cols_to_drop = None, cols_to_keep = None):
    self.drop_empty_cols = drop_empty_cols
    self.cols_to_drop = cols_to_drop
    self.cols_to_keep = cols_to_keep
    if (cols_to_keep is not None and cols_to_drop is not None):
      raise('Either remove columns or keep columns, do not do both.')
    return
  
  def fit(self, X, y = None):
    self.drop_cols_ = []    
    
    if self.drop_empty_cols:
      print(f'Columns {X.columns[X.isna().all()].to_list()} have no values. Will be dropped in transformation.')
      self.drop_cols_ = X.columns[X.isna().all()].to_list()
    
    if self.cols_to_drop is not None:
      self.drop_cols_ += self.cols_to_drop
    return self
  
  def transform(self, X, y = None): 
    X = cp.deepcopy(X)

    print(f'Dropping features {self.drop_cols_} due to identification during fit.')
    X = X.drop(columns = self.drop_cols_, errors = 'ignore')
    
    if self.cols_to_keep is not None:
      X = X.loc[:, self.cols_to_keep]
    
    return X

# COMMAND ----------

# MAGIC %md
# MAGIC #### class to scale values via multiple options

# COMMAND ----------

class MLRScaler(BaseEstimator, TransformerMixin):
  """
  This class will provide a host of options of scaling X and y variables in differing ways.
  
  Initialization: 
  
  y_col          : (default: 'y') the y value col
  scale_cols     : (default: None) the columns to scale. None will find all floating point variables and scale them. 
  scale_type     : (default: 'standard') the type of scale. valid values: 'standard', 'robust'. 
  scale_scope    : (default: 'universal') the scope of scaling. valid values: 'universal', 'categorical'
  categories     : (default: None) when scale_cope == 'categorical', this must be specified as a column or list of column to group by. 
  drop_y         : (default: True) when true, drop the Y value, but keep the scaling fit of it
  """
  
  def __init__(self, 
               y_col = 'y',
               scale_cols = None,
               scale_type = 'standard',
               scale_scope = 'universal',
               categories = None,
               drop_y = True):
    self.y_col       = y_col       
    self.scale_cols  = scale_cols  
    self.scale_type  = scale_type  
    self.scale_scope = scale_scope 
    self.categories  = categories  
    self.drop_y      = drop_y
    return
  
  def fit(self, X, y = None):
    X = cp.deepcopy(X)
    self.transformers_ = {}
    if (self.scale_scope == 'universal'):
      X['Category'] = 'universal'
      categories = ['Category']
    else:
      categories = self.categories
    
    # if no columns specified get the scaled values for all floating point columns
    if self.scale_cols is None:
      self.scale_cols_ = X.select_dtypes(['float', 'float32', 'float64']).columns.to_list()
    else:
      self.scale_cols_ = list(self.scale_cols)
    
    # if (self.y_col in self.scale_cols_): 
    #  self.scale_cols_.remove(self.y_col)
      
    print(f'Excluding {X.shape[0] - X.dropna().shape[0]} rows due to NaN for fitting purposes.')
    X = X.dropna().reset_index(drop = True)
    
    df_grouped = X.groupby(categories)
    
    def fit_group_scaler(Xg):
      self.transformers_[Xg.name] = {}
      for c in Xg[self.scale_cols_].columns.to_list():
        scaler = RobustScaler() if self.scale_type == 'robust' else StandardScaler()
        scaler.fit(Xg[[c]])
        self.transformers_[Xg.name][c] = scaler
      
    df_grouped.apply(fit_group_scaler)
    
    return self 
  
  def transform(self, X, y = None):
    X = cp.deepcopy(X)

    if (self.scale_scope == 'universal'):
      X['SCL_Category'] = 'universal'
      categories = ['SCL_Category']
    else:
      categories = self.categories
    
    print(f'Dropping {X.shape[0] - X.dropna().shape[0]} rows due to NaN values.')
    X = X.dropna().reset_index(drop = True)
    
    df_grouped = X.groupby(categories)
    
    def transform_group(Xg):
      for c in Xg.columns.to_list():
        if c in self.transformers_[Xg.name].keys(): 
          scaler = self.transformers_[Xg.name][c]
          Xg.loc[:, [c]] = scaler.transform(Xg.loc[:, [c]])
      return Xg
      
    # scale_cols = self.scale_cols_
    # X.loc[:, scale_cols] = df_grouped[scale_cols].apply(transform_group)
    X = df_grouped.apply(transform_group)
    
    X = X.drop(columns = 'SCL_Category', errors = 'ignore')
    
    # if (self.drop_y): 
    #  X = X.drop(columns = self.y_col, errors = 'ignore')
    return X 
  
  def fit_transform(self, X, y = None):
    print(f'received y [scaler]: {y}')
    self.fit(X, y)
    return self.transform(X, y)
  
  def inverse_transform(self, X, copy = None):
    X = cp.deepcopy(X)

    if (self.scale_scope == 'universal'):
      X['SCL_Category'] = 'universal'
      categories = ['SCL_Category']
    else:
      categories = self.categories
    
    df_grouped = X.groupby(categories)
    
    def transform_group(Xg):
      for c in Xg.columns.to_list():
        if c in self.transformers_[Xg.name].keys(): 
          scaler = self.transformers_[Xg.name][c]
          Xg.loc[:, [c]] = scaler.inverse_transform(Xg.loc[:, [c]])
      return Xg
      
    #scale_cols = self.scale_cols_ + ([self.y_col] if self.y_col in X.columns else [])
    #X.loc[:, scale_cols] = df_grouped[scale_cols].apply(transform_group)
    X = df_grouped.apply(transform_group)
    
    X = X.drop(columns = 'SCL_Category', errors = 'ignore')
    
    return X

# COMMAND ----------

tx_cs = MLRColumnSelector(cols_to_drop = p_exclude_cols)
df_tx_cs = tx_cs.fit_transform(df_raw)

tx_run = MLRRunTransitionCalculator(date_col = p_date_col, resin_col = p_resin_col, status_col = p_status_col, status_on_val = p_status_on_val)
df_tx_run = tx_run.fit_transform(df_tx_cs)

tx_mi = MLRMiHlmiCombiner(resin_col = p_resin_col, y_cols = p_y_cols, offspec_col = True, convert_hlmi_to_mi = True, spec_site = p_site, test_type_det = 'specs')
df_tx_mi = tx_mi.fit_transform(df_tx_run)

tx_dt = MLRCalculate_dS_dT(resin_col = p_resin_col, lag_cols = 'y', shifts = range(1, 6), y_col = None)
df_tx_dt = tx_dt.fit_transform(df_tx_mi)

# COMMAND ----------

# tx_scl_u = MLRScaler(scale_type = 'robust', scale_scope = 'universal', categories = ['Extruder', 'Test Type'], drop_y = False)
# df_tx_scl_u = tx_scl_u.fit_transform(df_tx_dt)

tx_scl_c = MLRScaler(scale_type = 'robust', scale_scope = 'categorical', categories = ['Extruder', 'Test Type'], drop_y = False)
df_tx_scl_c = tx_scl_c.fit_transform(df_tx_dt)
print(f'{df_tx_scl_c.shape}')

# COMMAND ----------

tx_scl_c.transformers_

# COMMAND ----------

df_tx_dt[[p_date_col, 'Die Plate Pressure', 'Extruder Fluff Feed Rate', 'y']].sample(n = 10, random_state = 123)

# COMMAND ----------

df_tx_scl_c[[p_date_col, 'Die Plate Pressure', 'Extruder Fluff Feed Rate', 'y']].sample(n = 10, random_state = 123)

# COMMAND ----------

df_inv_tx = tx_scl_c.inverse_transform(df_tx_scl_c[[p_date_col, 'Extruder', 'Test Type', 'Die Plate Pressure', 'Extruder Fluff Feed Rate', 'y']])
df_inv_tx[[p_date_col, 'Die Plate Pressure', 'Extruder Fluff Feed Rate', 'y']].sample(n = 10, random_state = 123)

# COMMAND ----------

p_extruder = '6-2'
df_cht = df_tx_scl_c
df_cht = df_cht[df_cht.Extruder.eq(p_extruder) & df_cht.dT.between(-12, 12)]

plt.figure(figsize = (20, 12))
fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 12))
sns.kdeplot(x = 'y',    hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[0, 0])
sns.kdeplot(x = 'd(y)', hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[0, 1])
sns.kdeplot(x = 'Gear Pump Bearing 1 Temp',    hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[1, 0])
sns.kdeplot(x = 'd(Gear Pump Bearing 1 Temp)', hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[1, 1])
sns.kdeplot(x   = 'Extruder Fluff Feed Rate',  hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[2, 0])
sns.kdeplot(x = 'd(Extruder Fluff Feed Rate)', hue = 'Test Type', fill = True, alpha = 0.4, data = df_cht, ax = ax[2, 1])


disp_cols = [p_date_col, p_resin_col, 'Test Type', 'y', 'y_prev', 'd(y)', 'dT', 'Gear Pump Bearing 1 Temp', 'd(Gear Pump Bearing 1 Temp)', 'Extruder Fluff Feed Rate', 'd(Extruder Fluff Feed Rate)']
df_cht[df_cht['d(Gear Pump Bearing 1 Temp)'].abs().gt(1)
       & df_cht['d(y)'].abs().gt(100)][disp_cols]



# COMMAND ----------

# MAGIC %md 
# MAGIC ### pipeline test of preprocessing

# COMMAND ----------

from sklearn import set_config
from sklearn.utils import parallel_backend
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from joblibspark import register_spark

# COMMAND ----------

df_raw.columns
X_cols_num = ['Gear Pump Bearing Average Temp', 'Extruder Fluff Feed Rate', 'Screenpack dP', 'Gear Pump Discharge Pressure', 'Gear Pump Amps']
X_cols_selection = ['Extruder', p_date_col, p_resin_col] + X_cols_num + p_y_cols
X_cols_group = ['Extruder', 'Test Type']
X_cols_cat = ['Extruder', 'Test Type']

# COMMAND ----------

dataset_pipeline = Pipeline(
  steps = [
    ('run_calc', MLRRunTransitionCalculator(resin_col = p_resin_col, status_col = p_status_col, status_on_val = p_status_on_val)),
    ('selector', MLRColumnSelector(cols_to_keep = X_cols_selection)),
    ('mi_hlmi_combiner', MLRMiHlmiCombiner(resin_col = p_resin_col, y_cols = p_y_cols, spec_site = p_site)),
    ('diff_lag', MLRCalculate_dS_dT(resin_col = p_resin_col, lag_cols = 'y', shifts = range(1, 6), y_col = 'y')),
  ], 
  verbose = True
)

set_config(display = 'diagram')
dataset_pipeline

# COMMAND ----------

df_trn = df_raw[df_raw.Extruder.eq('7-1') & df_raw[p_date_col].lt('2022-01-01')]
df_tst = df_raw[df_raw.Extruder.eq('7-1') & df_raw[p_date_col].ge('2022-01-01')]

df_trn = dataset_pipeline.fit_transform(df_trn)
df_tst = dataset_pipeline.transform(df_tst)

# COMMAND ----------



# COMMAND ----------

pipeline = Pipeline(
  steps = [
    ('run_calc', MLRRunTransitionCalculator(resin_col = p_resin_col, status_col = p_status_col, status_on_val = p_status_on_val)),
    ('selector', MLRColumnSelector(cols_to_keep = X_cols_selection)),
    ('mi_hlmi_combiner', MLRMiHlmiCombiner(resin_col = p_resin_col, y_cols = p_y_cols, spec_site = p_site)),
    ('diff_lag', MLRCalculate_dS_dT(resin_col = p_resin_col, lag_cols = 'y', shifts = range(1, 6), y_col = 'y')),
    ('scaler', MLRScaler(scale_type = 'robust', scale_scope = 'categorical', categories = X_cols_group, drop_y = True)),
    ('model_prep', 
     ColumnTransformer([
      ('split', 
       Pipeline(steps = [
          ('selector', MLRColumnSelector(cols_to_keep = X_cols_cat)),
          ('ohe', OneHotEncoder(handle_unknown = 'ignore', sparse=False))
       ]),
       X_cols_cat + [p_resin_col, p_date_col])
      ],
      remainder = 'passthrough'
     )),
    #('regressor', XGBRegressor())
  ], 
  verbose = True
)

set_config(display = 'diagram')
pipeline

# COMMAND ----------

# {'subsample': 0.7, 'n_estimators': 500, 'min_child_weight': 9, 'max_depth': 9, 'learning_rate': 0.1, 'colsample_bytree': 0.7}
grid_search = [{
  'regressor__subsample': [ 0.7 ],
  'regressor__n_estimators': [ 500 ],
  'regressor__min_child_weight': [ 9 ],
  'regressor__max_depth': [ 9 ],
  'regressor__learning_rate': [ 0.1 ],
  'regressor__colsample_bytree': [ 0.7 ],
  'scaler__scale_type': ['standard', 'robust'],
  'scaler__scale_scope': ['universal', 'categorical'],
}]

model = RandomizedSearchCV(
  pipeline, 
  param_distributions = grid_search, 
  cv = TimeSeriesSplit(n_splits = 10), 
  n_iter = 4,
  n_jobs = None,
  refit = True,
  scoring = 'neg_mean_absolute_percentage_error'
)

# COMMAND ----------

df_trn = df_raw[df_raw.Extruder.eq('7-1') & df_raw[p_date_col].lt('2022-01-01')]
df_tst = df_raw[df_raw.Extruder.eq('7-1') & df_raw[p_date_col].ge('2022-01-01')]

# COMMAND ----------

pipeline.fit(df_trn, 1)

# COMMAND ----------

from sklearn.utils import parallel_backend
from joblibspark import register_spark

register_spark()

with parallel_backend('spark', n_jobs = 32):
  model.fit(df_trn, df_train[y_col])

# COMMAND ----------

list(set(X_cols_num) - set(['y']))

# COMMAND ----------


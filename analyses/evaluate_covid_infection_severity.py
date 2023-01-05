# Author: Samantha Piekos
# Date: 11/22/22


# load environment
import pyspark.sql.functions as f
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import shap

from datetime import date, datetime, timedelta
from dateutil.relativedelta import *
from mpl_toolkits import mplot3d
from pyspark.sql.functions import col
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.graphics.gofplots import qqplot
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.utils import shuffle
from statsmodels.stats.proportion import proportion_confint

dbutils.library.installPyPI('shap',version="0.37.0")
dbutils.library.installPyPI("pywaffle")
dbutils.library.installPyPI("rpy2")
from pywaffle.waffle import Waffle
import rpy2

np.random.seed(611)
%matplotlib inline
spark.sql("SET spark.databricks.delta.formatCheck.enabled=false")


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
def determine_covid_days_from_conception(row):  # identify the number of days into pregnancy the first covid positive test was observed
  count = 0
  delivery_date = row['ob_delivery_delivery_date']
  gestational_days = row['gestational_days']
  check_date = datetime.datetime(2021, 2, 14, 0, 0, 0)
  for item in row['ordering_datetime_collect_list']:
    if row['result_short_collect_list'][count] == 'positive' and (check_date - item).total_seconds() > 0:
      days_from_conception = row['gestational_days'] - (delivery_date - item)/np.timedelta64(1, 'D')
      if 0 <= days_from_conception <= row['gestational_days']:
        return(days_from_conception, item)
    count += 1
  return(None, None)


def determine_covid_maternity_cohorts(df):
  import pandas as pd
  import numpy as np
  import datetime
  df = df.toPandas()
  cols = df.columns
  df_maternity_all = pd.DataFrame(columns=cols)
  df_temp = pd.DataFrame(columns=['pat_id', 'covid_test_date', 'covid_test_number_of_days_from_conception'])
  list_gestational_days_at_covid_infection = []
  list_positive_covid_test_order_date = []
  for index, row in df.iterrows():
    if row['number_of_fetuses'] == 1 and row['gestational_days'] >= 140:
      days_from_conception, order_date = determine_covid_days_from_conception(row)
      if days_from_conception is not None:
        df_maternity_all = df_maternity_all.append(row)
        df_temp = df_temp.append({'pat_id': row['pat_id'], 'covid_test_number_of_days_from_conception': days_from_conception, 'covid_test_date': order_date}, ignore_index=True)
  print('Number of women with covid infections during a completed pregnancy: ' + str(len(df_maternity_all.pat_id.unique())))
  return(df_maternity_all, df_temp.drop_duplicates())


def medication_orders_from_first_covid_positive(cohort): 
  """From the provided dataframe, return all medication orders 2 weeks prior to 8 weeks after to the first covid positive date during pregnancy 
  Parameters:
  cohort_df (PySpark df): A dataframe containing 'pat_id', 'gestational_days', 'ob_delivery_delivery_date', 'positive_datetime_min', 'positive_datetime_lmp_min'
  
  Returns:
  PySpark df: medication order from first covid positive date to date of delivery - 2 
  """
  filter_string = \
  """
  start_date >= date_sub(covid_test_date, 14) AND
  start_date < date_add(covid_test_date, 56)
  """
  list_administered = ['Anesthesia Volume Adjustment', 'Bolus from Bag', 'Continued Bag', 'Dispense to Home', 'Given', 'Given by Other', 'New Bag', 'New Syringe/Cartridge', 'Patch Applied', 'Push', 'Rate Verify', 'Rate Change', 'Rate/Dose Change-Dual Sign', 'Rate/Dose Verify-Dual Sign', 'Restarted']
  keep_cohort_columns = ['pat_id', 'gestational_days', 'ob_delivery_delivery_date', 'covid_test_date']
  cohort_concise = cohort.select(keep_cohort_columns)
  meds = spark.sql("SELECT * FROM rdp_phi_sandbox.hadlock_medication_admin_start_2008")
  medorders = cohort_concise.join(meds, ['pat_id'], 'left').filter(filter_string).where(f.col('action_taken').isin(list_administered))
  return medorders


def order_med_counts(med_df):
  """From the provided dataframe, return counts of inpatient and outpatient medication order  
  Parameters:
  cohort_df (PySpark df): A dataframe containing 'pat_id' 'instance' 'lmp' 'order_mode'
  
  Returns:
  PySpark df: count of inpatient and outpatient medication order 
  """
  ordermode_count = med_df.groupBy(['pat_id']).pivot("order_mode").count().withColumnRenamed('Inpatient', 'inpatient_meds_count').withColumnRenamed('Outpatient', 'outpatient_meds_count')
  ordermode_count = ordermode_count.na.fill(value=0)
  return ordermode_count


def unique_ingredient_counts(med_df):
  """From the provided dataframe, return counts of inpatient and outpatient medication order  
  Parameters:
  cohort_df (PySpark df): A dataframe containing 'pat_id' 'instance' 'lmp' 'medication_description' 'term_type'
  
  Returns:
  PySpark df: count of unique ingredient 
  """
  #meds_ingredient = med_df.withColumn("new", F.arrays_zip('medication_description', 'term_type'))\
   #    .withColumn("new", F.explode("new"))\
    #   .select(*med_df.columns, F.col("new.medication_description").alias("ingredient"), F.col("new.term_type").alias("type")).filter(F.col('type') == 'Ingredient')
  ingredient_count = meds_ingredient.groupBy(['pat_id']).agg(f.countDistinct("short_name")).withColumnRenamed('count(short_name)', 'unique_inpatient_medication_count')
  return ingredient_count


def consolidate_race_responses(l):
  l_new = []
  for i in l:
    if i == 'White':
      l_new.append('White or Caucasian')
    elif i == 'Patient Refused' or i == 'Unable to Determine' or i == 'Declined' or i == 'Unknown':
      continue
    else:
      l_new.append(i)
  l_new = list(set(l_new))
  return(l_new)


def handle_multiracial_exceptions(l):
  l_new = consolidate_race_responses(l)
  if l_new is None:
    return('Unknown')
  if len(l_new) == 1:
    return(l_new[0])
  if 'Other' in l_new:
    l_new.remove('Other')
    if l_new is None:
      return('Other')
    if len(l_new) == 1:
      return(l_new[0])
  return('Multiracial')


def format_race(i):
  if i is None:
    return('Unknown')
  if i == 0:
    return('Unknown')
  if len(i) > 1:
    return('Multiracial')
  if i[0] == 'White':
    return('White or Caucasian')
  if i[0] == 'Declined' or i[0] == 'Patient Refused':
    return('Unknown')
  if i[0] == 'Unable to Determine':
    return('Unknown')
  return(i[0])


def format_ethnic_group(i):
  if i is None:
    return('Unknown')
  if i == 'American' or i == 'Samoan':
    return('Not Hispanic or Latino')
  if i == 'Patient Refused' or i == 'None':
    return('Unknown')
  return(i)


def make_class_encoder_dict(l):
  d = {}
  c = 0
  for i in l:
    d[i] = c
    c += 1
  return(d)


def format_parity(i):
  if i is None:
    return 'Unknown'
  i = int(i)
  if i == 0 or i == 1:
    return 'Nulliparity'
  if i > 1 and i < 5:
    return 'Low Multiparity'
  return 'Grand Multipara'


def format_gravidity(gravidity):
  if gravidity is None:
    return 'Unknown'
  else:
    gravidity = int(gravidity)
    if gravidity == 0 or gravidity == 1:
      return 'Primagravida'
    elif gravidity > 1 and gravidity < 6:
      return 'Low Multigravidia'
    else:
      return 'Grand Multigravidia'

    
def format_preterm_history(i, gestational_days):
  if i is None:
    return -1
  i = int(i)
  if i == 0 or (i == 1 or gestational_days < 259):
    return 0
  else:
    return 1
  return -1
    
  
def encode_classes(df):
  dict_fetal_sex = make_class_encoder_dict(df.ob_hx_infant_sex.unique())
  dict_insurance = make_class_encoder_dict(df.insurance.unique())
  return(dict_fetal_sex, dict_insurance)


def encode_delivery_method(i):
  '''
  0 = Vaginal
  1 = C-Section
  -1 = Unknown
  '''
  list_vaginal = ['Vaginal, Spontaneous',
       'Vaginal, Vacuum (Extractor)',
       'Vaginal, Forceps', 'Vaginal < 20 weeks',
       'Vaginal, Breech', 'VBAC, Spontaneous',
       'Vaginal Birth after Cesarean Section',
       'Spontaneous Abortion']
  list_c_section = ['C-Section, Low Transverse',
       'C-Section, Low Vertical',
       'C-Section, Classical',
       'C-Section, Unspecified']
  if i in list_vaginal:
    return(0)
  if i in list_c_section:
    return(1)
  return(-1)


def format_delivery_cord_vessels(i):
  if i is None or i == 'Unknown':
    return -1
  elif i == 0 or i == 1:
    return -1
  elif i == '2 Vessels':
    return 0
  elif i == '3 Vessels':
    return 1
  else:
    print('Error in formatting cord vessels: ' + i)
  return(-1)


def format_dataframe_for_modeling(df):
  dict_race = {'White or Caucasian': 0, 'Unknown': -1, 'Asian': 1, 'Multiracial': 6, 'Other': 2, 'Black or African American': 3,\
               'Native Hawaiian or Other Pacific Islander': 4, 'American Indian or Alaska Native': 5}
  dict_ethnic_groups = {'Unknown': -1, 'Hispanic or Latino': 1, 'Not Hispanic or Latino': 0, 0: -1, 'Hmong': 0}
  dict_fetal_sex = {None: -1, 'Male': 1, 'Female': 0, 'Other': -1, 'Unknown': -1, 0: -1}
  dict_insurance = {'Medicaid': 0, 'Medicare': 0, 'Commercial': 1, None: 0, 'None': 0, 0: 0, 'Other': 0, 'Uninsured-Self-Pay': 0}
  dict_parity = {'Unknown': -1, 'Nulliparity': 0, 'Low Multiparity': 1, 'Grand Multipara': 2} 
  dict_gravidity = {'Unknown': -1, 'Primagravida': 0, 'Low Multigravidia': 1, 'Grand Multigravidia': 2}
  for index, row in df.iterrows():
    df.at[index, 'race'] = format_race(row['race'])
    df.at[index, 'ethnic_group'] = format_ethnic_group(row['ethnic_group'])
    df.at[index, 'Parity'] = format_parity(row['Parity'])
    df.at[index, 'Gravidity'] = format_gravidity(row['Gravidity'])
    df.at[index, 'Preterm_history'] = format_preterm_history(row['Preterm_history'], row['gestational_days'])
    df.at[index, 'delivery_cord_vessels'] = format_delivery_cord_vessels(row['delivery_cord_vessels'])
  for index, row in df.iterrows():
    df.at[index, 'race'] = dict_race[row['race']]
    df.at[index, 'ethnic_group'] = dict_ethnic_groups[row['ethnic_group']]
    df.at[index, 'ob_hx_infant_sex'] = dict_fetal_sex[row['ob_hx_infant_sex']]
    df.at[index, 'insurance'] = dict_insurance[row['insurance']]
    df.at[index, 'Parity'] = dict_parity[row['Parity']]
    df.at[index, 'Gravidity'] = dict_parity[row['Parity']]
    df.at[index, 'delivery_delivery_method'] = encode_delivery_method(row['delivery_delivery_method'])
  df['pregravid_bmi'].fillna(df['pregravid_bmi'].median(), inplace=True)
  df['pregravid_bmi'] = pd.to_numeric(df['pregravid_bmi'], downcast="float")
  return(df)


def count_diagnoses(df):
  d_total = {}
  d_unique = {}
  d_seen = {}
  for index, row in df.iterrows():
    k = row['pat_id']
    diagnosis = row['dx_id']
    if k not in d_seen:
      d_seen[k] = [diagnosis]
      d_total[k] = 1
      d_unique[k] = 1
    elif diagnosis not in d_seen[k]:
      d_unique[k] += 1
      d_total[k] += 1
      d_seen[k].append(diagnosis)
    else:
      d_total[k] += 1
  return(d_total, d_unique)


def add_diagnoses_counts(df_diagnoses, df):
  d_total, d_unique = count_diagnoses(df_diagnoses)
  df_temp = pd.DataFrame(columns=['pat_id', 'n_total_diagnoses', 'n_unique_diagnoses'])
  for k, v in d_total.items():
    df_temp = df_temp.append({'pat_id': k, 'n_total_diagnoses': v, 'n_unique_diagnoses': d_unique[k]}, ignore_index=True)
  df_final = df.merge(df_temp, left_on='pat_id', right_on='pat_id', how='left')
  df_final = df_final.fillna(0)
  return(df_final)


def count_medications(df):
  d_total = {}
  d_unique = {}
  d_seen = {}
  for index, row in df.iterrows():
    k = row['pat_id']
    medication = row['count(ingredient)']
    if k not in d_seen:
      d_seen[k] = [medication]
      d_total[k] = 1
      d_unique[k] = 1
    elif diagnosis not in d_seen[k]:
      d_unique[k] += 1
      d_total[k] += 1
      d_seen[k].append(medication)
    else:
      d_total[k] += 1
  return(d_total, d_unique)


def add_medications_counts(df_medications, df):
  d_total, d_unique = count_medications(df_medications)
  df_temp = pd.DataFrame(columns=['pat_id', 'n_total_medications', 'n_unique_medications'])
  for k, v in d_total.items():
    df_temp = df_temp.append({'pat_id': k, 'n_total_medications': v, 'n_unique_medications': d_unique[k]}, ignore_index=True)
  df_final = df.merge(df_temp, left_on='pat_id', right_on='pat_id', how='left')
  df_final = df_final.fillna(0)
  return(df_final)


def count_encounters(df):
  d = {}
  for index, row in df.iterrows():
    k = row['pat_id']
    if k not in d:
      d[k] = 1
    else:
      d[k] += 1
  return(d)


def add_max_to_dict(d, k, v):
  if k not in d:
    d[k] = v
  elif v > d[k]:
    d[k] = v
  return(d)


def check_high_flow_o2_plus_vasopressor(oxygen_device, vasopressor_sum):
  if math.isnan(vasopressor_sum):
    return oxygen_device
  vasopressor_sum = float(vasopressor_sum)
  if oxygen_device == 2 and vasopressor_sum > 0:
    return 3
  return oxygen_device


def format_oxygen_device(df):
  '''
  0 = None
  1 = Low-Flow Oxygen Device
  2 = High-Flow Oxygen Device
  3 = Venhilator
  '''
  dict_conversion = {
 'non-invasive ventilation (i.e. bi-level)': 3,
 'T-piece': 3,
 'heated': 2,
 'high-flow nasal cannula': 2,
 'manual bag ventilation': 2,
 'nonrebreather mask': 2,
 'partial rebreather mask': 2,
 'face tent': 2,
 'aerosol mask': 1,
 'nasal cannula': 1,
 'nasal cannula with humidification': 1,
 'nasal cannula with reservoir (i.e. oximizer)': 1,
 'simple face mask': 1,
 'open oxygen mask': 1,
 'blow by': 1,
 'other (see comments)': 1,
 'room air': 0,
 'Room air (None)': 0}
  dict_oxygen_device_type = {}
  for index, v in df.iterrows():
    k = v.pat_id
    item = v.oxygen_device_concat_ws
    dict_oxygen_device_type = add_max_to_dict(dict_oxygen_device_type, k, 0)
    if item is None:
      continue
    for i in item.split(';'):
      if i is None:
        dict_oxygen_device_type = add_max_to_dict(dict_oxygen_device_type, k, 0)
        continue
      i = dict_conversion[i]
      dict_oxygen_device_type = add_max_to_dict(dict_oxygen_device_type, k, i)
    oxygen_device = dict_oxygen_device_type[k]
    vasopressor_sum = v.vasopressor_sum
    dict_oxygen_device_type[k] = check_high_flow_o2_plus_vasopressor(oxygen_device, vasopressor_sum)
  return dict_oxygen_device_type
  
  
def format_patient_class(df):
  '''
  0 = None
  1 = Outpatient
  2 = Urgent Care / Emergency
  3 = Inpatient
  4 = ICU
  '''
  dict_conversion = {'Emergency': 2,
 'Extended Hospital Outpatient': 1,
 'Free Standing Outpatient': 1,
 'Hospital Ambulatory Surgery': 1,
 'Infusion Series': 1,
 'Inpatient': 3,
 'Inpatient Rehab Facility': 3,
 'MU Urgent Care': 2,
 'NM Series': 1,
 'Observation': 1,
 'Other Series': 1,
 'Outpatient': 1,
 'Outreach': 1,
 'Radiation/Oncology Series': 1,
 'Rural Health': 1,
 'Specimen': 1,
 'Therapies Series': 1}
  dict_patient_class = {}
  for index, v in df.iterrows():
    k = v.pat_id
    l = v.patient_class_collect_list
    dict_patient_class = add_max_to_dict(dict_patient_class, k, 0)
    if l is None:
      continue
    for item in l:
      if item is None:
        continue
      item = dict_conversion[item]
      add_max_to_dict(dict_patient_class, k, item)
  return(dict_patient_class)


def format_base_patient_class(df):
  '''
  0 = None
  1 = Outpatient
  2 = Urgent Care / Emergency
  3 = Inpatient
  '''
  dict_conversion = {'None': 0,
 'Outpatient': 1,
 'Emergency': 2,
 'Inpatient': 3}
  dict_patient_class = {}
  for index, v in df.iterrows():
    k = v.pat_id
    l = v.base_patient_class_collect_list
    dict_patient_class = add_max_to_dict(dict_patient_class, k, 0)
    if l is None:
      continue
    for item in l:
      if item is None:
        continue
      item = dict_conversion[item]
      add_max_to_dict(dict_patient_class, k, item)
  return(dict_patient_class)


def add_patient_deaths(df):
  # add patient deaths that occurred within 3 months after a covid infection
  df.createOrReplaceTempView("temp")
  df_patient = spark.sql("SELECT * FROM rdp_phi.patient")
  df_patient.createOrReplaceTempView("patient")
  df_plus_death = spark.sql(
    """
    FROM patient AS p
    RIGHT JOIN temp AS t
    ON p.pat_id = t.pat_id
      AND p.deathdate - interval '3' month < t.covid_test_date
    SELECT t.*, p.deathdate
    """)
  return df_plus_death


def add_column_to_df(df, dict_data, column_name):
  df[column_name] = df['pat_id'].map(dict_data)
  return df


def format_encounters(df_encounters, df):
  dict_n_encounters = count_encounters(df_encounters)
  df_encounters['who_score'] = df_encounters['who_score'].fillna(3)
  dict_oxygen_device = format_oxygen_device(df_encounters)
  # dict_patient_class = format_patient_class(df_encounters)
  dict_patient_class = format_base_patient_class(df_encounters)
  df_encounters = df_encounters.sort_values(['who_score',  'encounter_duration'], ascending=False).drop_duplicates('pat_id').sort_index()
  df_encounters = df_encounters.drop(columns=['oxygen_device_concat_ws', 'patient_class_collect_list'])
  df_encounters = add_column_to_df(df_encounters, dict_oxygen_device, 'max_oxygen_device')
  df_encounters = add_column_to_df(df_encounters, dict_patient_class, 'max_patient_class')
  df_encounters = add_column_to_df(df_encounters, dict_n_encounters, 'n_covid_encounters')
  df_final = df.merge(df_encounters, left_on='pat_id', right_on='pat_id', how='left')
  df_final['n_covid_encounters'] = df_final['n_covid_encounters'].fillna(0)
  df_final['max_oxygen_device'] = df_final['max_oxygen_device'].fillna(0)
  df_final['max_patient_class'] = df_final['max_patient_class'].fillna(0)
  df_final['who_score'] = df_final['who_score'].fillna(2)
  return df_final


def sum_list(l):
  l = list(l)
  s = 0
  for i in l:
    s += i
  return s


def calc_fishers_exact_test_not_simulated(dict_obs, dict_exp):
  import numpy as np
  import rpy2.robjects.numpy2ri
  from rpy2.robjects.packages import importr
  rpy2.robjects.numpy2ri.activate()

  stats = importr('stats')
  list_obs, list_exp = [], []
  total_obs = sum_list(dict_obs.values())
  total_exp = sum_list(dict_exp.values())
  for key in dict_obs.keys():
    list_obs.append(dict_obs[key])
    list_exp.append(dict_exp[key])
  list_contingency = np.array([list_obs, list_exp])
  print(list_contingency)
  res = stats.fisher_test(x=list_contingency)
  print(res) 
  
  
def calc_fishers_exact_test_2x2(df_1, df_2, k):
  import numpy as np
  import rpy2.robjects.numpy2ri
  from rpy2.robjects.packages import importr
  rpy2.robjects.numpy2ri.activate()

  stats = importr('stats')
  list_1, list_2 = [], []
  df_1_n = len(df_1[k])
  df_2_n = len(df_2[k])
  df_1_sum = sum(df_1[k])
  list_1.append(sum(df_1[k]))
  list_1.append(len(df_1[k])-df_1_sum)
  df_2_sum = sum(df_2[k])
  list_2.append(sum(df_2[k]))
  list_2.append(len(df_2[k])-df_2_sum)
  list_contingency = np.array([list_1, list_2])
  print(list_contingency)
  res = stats.fisher_test(x=list_contingency)
  print(res)
  

def make_covid_death_dict(df):
  d = {'Alive': 0, 'Dead': 0}
  for index, row in df.iterrows():
    if isinstance(row['deathdate'], datetime.datetime):
      d['Dead'] += 1
    else:
      d['Alive'] += 1
  print(d)
  return d


def make_covid_severity_dict(df):
  d = {'Mild': 0, 'Moderate': 0, 'Severe': 0}
  for index, row in df.iterrows():
    if isinstance(row['deathdate'], datetime.datetime):
      d['Severe'] += 1
    elif row['max_oxygen_device'] >= 2:
      d['Severe'] += 1
    elif row['max_patient_class'] >= 3 or row['max_oxygen_device'] == 1:
      d['Moderate'] += 1
    else:
      d['Mild'] += 1
  print(d)
  return d


def make_oxygen_device_dict(l):
  d = {'None': 0, 'Low-Flow': 0, 'High-Flow': 0, 'Venhilator': 0}
  for i in l:
    if i is None:
      d['None'] += 1
    elif i == -1:
      d['None'] += 1
    elif i == 0:
      d['None'] += 1
    elif i == 1:
      d['Low-Flow'] += 1
    elif i == 2:
      d['High-Flow'] += 1
    elif i == 3:
      d['Venhilator'] += 1
    else:
      print(i)
  print(d)
  return d


def make_vasopressor_dict(l):
  d = {'Yes': 0, 'No': 0}
  for i in l:
    if isinstance(i, float):
      i = float(i)
      if i > 0:
        d['Yes'] += 1
      else:
        d['No'] += 1
    else:
      d['No'] += 1
  print(d)
  return d


def make_oxygen_assistance_dict(d):
  new_d = {}
  #{'None': 0, 'Low-Flow': 0, 'High-Flow': 0, 'Venhilator': 0}
  new_d['None'] =  d['None']
  new_d['Oxygen Assistance'] = d['Low-Flow'] + d['High-Flow'] + d['Venhilator']
  print(new_d)
  return new_d


def make_patient_class_dict(df):
  d = {'None': 0, 'Outpatient': 0, 'Emergency Care': 0, 'Inpatient': 0}
  for index, row in df.iterrows():
    patient_class = row['max_patient_class']
    if patient_class < 1:
      d['None'] += 1
    elif patient_class == 1:
      d['Outpatient'] += 1
    elif patient_class == 2:
      d['Emergency Care'] += 1
    else:
      d['Inpatient'] += 1
  print(d)
  return d


def determine_inpatient_status(df):
  d = {'No': 0, 'Yes': 0}
  for index, row in df.iterrows():
    patient_class = row['max_patient_class']
    if patient_class >= 3:
      d['Yes'] += 1
    else:
      d['No'] += 1
  print(d)
  return d


def determine_hospitalization_status(df):
  d = {'No': 0, 'Yes': 0}
  for index, row in df.iterrows():
    patient_class = row['max_patient_class']
    if patient_class >= 2:
      d['Yes'] += 1
    else:
      d['No'] += 1
  print(d)
  return d


def make_inpatient_dict(df):
  d = {'Not Inpatient': 0, 'Inpatient': 0}
  for index, row in df.iterrows():
    patient_class = row['max_patient_class']
    if patient_class >= 3:
      d['Inpatient'] += 1
    else:
      d['Not Inpatient'] += 1
  print(d)
  return d


def make_hospitalization_dict(df):
  d = {'Not Hospitalized': 0, 'Hospitalized': 0}
  for index, row in df.iterrows():
    patient_class = row['max_patient_class']
    if patient_class >= 2:
      d['Hospitalized'] += 1
    else:
      d['Not Hospitalized'] += 1
  print(d)
  return d


def inpatient_medication_count(medorders):
  filter_string_inpatient = \
  """
  start_date >= date_sub(covid_test_date, 14) AND
  start_date < date_add(covid_test_date, 56) AND
  order_mode == 'Inpatient'
  """
  list_administered = ['Anesthesia Volume Adjustment', 'Bolus', 'Bolus from Bag', 'Calc Rate', 'Continued by Anesthesia', 'Continued Bag', 'Continue bag from transfer', 'Dispense to Home', 'Given', 'Given by Other', 'Given During Downtime', 'Milk Verified', 'New Bag', 'New Syringe/Cartridge', 'Patch Applied', 'Push', 'Rate Verify', 'Rate Change', 'Rate/Dose Change-Dual Sign', 'Rate/Dose Verify-Dual Sign', 'Restarted', 'Started During Downtime', 'Unheld by Provider']
  medorders_inpatient = medorders.filter(filter_string_inpatient).where(f.col('action_taken').isin(list_administered))
  medorders_inpatient = medorders_inpatient.groupBy(['pat_id']).agg(f.countDistinct("short_name")).withColumnRenamed('count(short_name)', 'unique_inpatient_medication_count')
  return medorders_inpatient


def outpatient_medication_count(medorders):
  filter_string_outpatient_start = \
  """
  start_date >= date_sub(covid_test_date, 14) AND
  start_date < date_add(covid_test_date, 56) AND
  order_mode == 'Outpatient'
  """
  
  filter_string_outpatient_end = \
  """
  end_date >= date_sub(covid_test_date, 14) AND
  end_date < date_add(covid_test_date, 56) AND
  order_mode == 'Oupatient'
  """
  medorders_outpatient_start = medorders.filter(filter_string_outpatient_start)
  medorders_outpatient_end = medorders.filter(filter_string_outpatient_end)
  medorders_outpatient = medorders_outpatient_start.union(medorders_outpatient_end).dropDuplicates(['pat_id', 'medication_id', 'pat_enc_csn_id'])
  medorders_outpatient = medorders_outpatient.groupBy(['pat_id']).agg(f.countDistinct("short_name")).withColumnRenamed('count(short_name)', 'unique_outpatient_medication_count')
  return medorders_outpatient


def total_medication_count(medorders):
  filter_string_inpatient = \
  """
  start_date >= date_sub(covid_test_date, 14) AND
  start_date < date_add(covid_test_date, 56) AND
  order_mode == 'Inpatient'
  """
  
  filter_string_outpatient_start = \
  """
  start_date >= date_sub(covid_test_date, 14) AND
  start_date < date_add(covid_test_date, 56) AND
  order_mode == 'Outpatient'
  """
  
  filter_string_outpatient_end = \
  """
  end_date >= date_sub(covid_test_date, 14) AND
  end_date < date_add(covid_test_date, 56) AND
  order_mode == 'Oupatient'
  """

  list_administered = ['Anesthesia Volume Adjustment', 'Bolus', 'Bolus from Bag', 'Calc Rate', 'Continued by Anesthesia', 'Continued Bag', 'Continue bag from transfer', 'Dispense to Home', 'Given', 'Given by Other', 'Given During Downtime', 'Milk Verified', 'New Bag', 'New Syringe/Cartridge', 'Patch Applied', 'Push', 'Rate Verify', 'Rate Change', 'Rate/Dose Change-Dual Sign', 'Rate/Dose Verify-Dual Sign', 'Restarted', 'Started During Downtime', 'Unheld by Provider']
  medorders_inpatient = medorders.filter(filter_string_inpatient).where(f.col('action_taken').isin(list_administered))
  medorders_outpatient_start = medorders.filter(filter_string_outpatient_start)
  medorders_outpatient_end = medorders.filter(filter_string_outpatient_end)
  medorders_all = medorders_inpatient.union(medorders_outpatient_start.union(medorders_outpatient_end)).dropDuplicates(['pat_id', 'medication_id', 'pat_enc_csn_id'])
  medorders_all = medorders_all.groupBy(['pat_id']).agg(f.countDistinct("short_name")).withColumnRenamed('count(short_name)', 'unique_medication_count')
  return medorders_all


def combine_medication_counts(medorders_inpatient, medorders_outpatient, medorders_all):
  medorders_inpatient.createOrReplaceTempView("inpatient")  
  medorders_outpatient.createOrReplaceTempView("outpatient")
  temp = spark.sql(
  """
    FROM inpatient AS i
    INNER JOIN outpatient AS o
    ON i.pat_id = o.pat_id
    SELECT i.*, o.unique_outpatient_medication_count
  """)
  medorders_all.createOrReplaceTempView("meds")
  temp.createOrReplaceTempView("temp") 
  medorders_final = spark.sql(
  """
    FROM temp AS t
    INNER JOIN meds AS m
    ON t.pat_id = m.pat_id
    SELECT t.*, m.unique_medication_count
  """)
  return medorders_final
  

def unique_medication_counts(cohort): 
  """From the provided dataframe, return all unique inpatient medication orders administered 2 weeks prior to 8 weeks after to the first covid positive date during pregnancy 
  Parameters:
  cohort_df (PySpark df): A dataframe containing 'pat_id', 'gestational_days', 'ob_delivery_delivery_date', 'positive_datetime_min', 'positive_datetime_lmp_min'
  
  Returns:
  PySpark df: medication order from first covid positive date to date of delivery - 2 
  """
  medorders = get_medication_orders(cohort_df=cohort.select(['pat_id', 'instance', 'episode_id', 'covid_test_date']), join_table='medicationadministration', omit_columns=None)
  print('Adding inpatient unique medication count...')
  medorders_inpatient = inpatient_medication_count(medorders)
  print('Adding outpatient unique medication count...')
  medorders_outpatient = outpatient_medication_count(medorders)
  print('Adding all unique medication count...')
  medorders_all = total_medication_count(medorders)
  medorders_final = combine_medication_counts(medorders_inpatient, medorders_outpatient, medorders_all)
  print('Unique medication counts added!')
  print('\n')
  return medorders_final


def run_mann_whitney_u_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.mannwhitneyu(data_1, data_2))


def expand_med_arrays(df):
  df.createOrReplaceTempView("temp")
  temp_2 = spark.sql("""
  SELECT med.medication_id_collect_set AS medication_id, med.name_collect_set AS name, med.pat_enc_csn_id_collect_set AS pat_enc_csn_id, pat_id
  FROM (SELECT DISTINCT explode(arrays_zip(medication_id_collect_set, name_collect_set, pat_enc_csn_id_collect_set)) AS med, pat_id FROM temp)
  ORDER BY pat_id
  """)
  
  temp_2.createOrReplaceTempView("temp2")
  df_formatted = spark.sql("""
  FROM temp AS m
  INNER JOIN temp2 AS t
  ON m.pat_id = t.pat_id
  SELECT m.pat_id, m.instance, m.episode_id, m.covid_test_date, m.birth_date, m.death_date, m.sex, m.ethnic_group, m.race, m.time_index_day, m.time_index_day_label, t.medication_id, t.name, element_at(array_distinct(m.order_mode_collect_set), 1) AS order_mode, element_at(array_distinct(m.action_taken_collect_set), 1) AS action_taken, t.pat_enc_csn_id, m.vasopressor_sum
  """).dropDuplicates(['pat_id', 'medication_id', 'pat_enc_csn_id'])
  return df_formatted


def plot_waffle_chart_inpatient(data):
	colors = {'None': 'forestgreen', 'Outpatient': 'lightskyblue', 'Emergency Care': 'mediumorchid', 'Inpatient': 'orangered'}
	list_colors = []
	list_data = []
	list_order = ['None', 'Outpatient', 'Emergency Care', 'Inpatient']
	for key in list_order:
		list_colors.append(colors[key])
		list_data.append((data[key]/sum(data.values())*100))
	fig = plt.figure(
		FigureClass=Waffle,
		rows=10,
		values=list_data,
		colors=list_colors,
		legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)}
		)
	plt.show()
    
    
def extract_matched_data_from_unmatched_df(df_matched, df):
  df_matched['id'] = df_matched['pat_id'].astype(str) + df_matched['episode_id'].astype('int').astype('str') + df_matched['child_episode_id'].astype('int').astype('str')
  df['id'] = df['pat_id'].astype(str) + df['episode_id'].astype('int').astype('str') + df['child_episode_id'].astype('int').astype('str')
  df_matched_final = pd.DataFrame(columns=df.columns)
  for index, row in df_matched.iterrows():
    item = df.loc[(df['id'] == row['id'])]
    df_matched_final = pd.concat([df_matched_final, item], ignore_index=True, verify_integrity=False)
    #df_matched_final = df_matched_final.append(item, ignore_index=True)
  df_matched_final.drop('id', axis=1, inplace=True)
  return df_matched_final


  # load data
  df_breakthrough_jj_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_jj AS jj WHERE jj.covid_test_date >= '2021-12-25'")
print('# Number of women with breakthrough covid infection (J&J) during pregnancy who has delivered: ' + str(df_breakthrough_jj_pyspark.count()))

df_breakthrough_mrna_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_mrna AS mrna WHERE mrna.covid_test_date >= '2021-12-25'")
print('# Number of women with breakthrough covid infection during pregnancy who has delivered: ' + str(df_breakthrough_mrna_pyspark.count()))

df_breakthrough_moderna_pyspark = df_breakthrough_mrna_pyspark.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of women with breakthrough covid infection (Moderna) during pregnancy who has delivered: ' + str(df_breakthrough_moderna_pyspark.count()))

df_breakthrough_pfizer_pyspark = df_breakthrough_mrna_pyspark.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of women with breakthrough covid infection (Pfizer) during pregnancy who has delivered: ' + str(df_breakthrough_pfizer_pyspark.count()))

df_breakthrough_one_mrna_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_covid_infection AS one_mrna WHERE one_mrna.covid_test_date >= '2021-12-25'")
print('# Number of women with breakthrough covid infection (one mRNA shot) during pregnancy who has delivered: ' + str(df_breakthrough_one_mrna_pyspark.count()))

df_covid_immunity_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_unvaccinated_covid_immunity AS ci WHERE ci.covid_test_date >= '2021-12-25'")
print('# Number of women with covid-induced immunity prior to pregnancy that got a subsequent COIVD-19 infection while unvaccinated during pregnancy who has delivered: ' + str(df_covid_immunity_pyspark.count()))

df_unvaccinated_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_unvaccinated AS ucm WHERE ucm.ob_delivery_delivery_date >= '2021-01-26' AND ucm.covid_test_date >= '2021-12-25'")
print('# Number of pregnant women unvaccinated at time of COIVD-19 infection during pregnancy who have delivered after 1-25-21: ' + str(df_unvaccinated_pyspark.count()))

df_unvaccinated_matched_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_matched_control_covid AS ucmc WHERE ucmc.ob_delivery_delivery_date >= '2021-01-26' AND ucmc.covid_test_date >= '2021-12-25'")
#df_unvaccinated_pyspark.createOrReplaceTempView("unvaccinated")
#df_unvaccinated_matched_pyspark .createOrReplaceTempView("unvaccinated_matched")
#df_unvaccinated_matched_pyspark  = spark.sql(
#"""
#SELECT u.*
#FROM unvaccinated AS u
#INNER JOIN unvaccinated_matched AS um 
#ON u.pat_id = um.pat_id
#  AND u.instance = um.instance
#  AND u.episode_id = um.episode_id
#  AND u.child_episode_id = um.child_episode_id
#  """)
print('# Number of pregnant women in unvaccinated matched control with a covid infection during pregnancy: ' + str(df_unvaccinated_matched_pyspark.count()))

df_breakthrough_two_shots_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted_covid_breakthrough AS two_shots WHERE two_shots.covid_test_date >= '2021-12-25'")
print('# Number of women with only 2 shots whose last shot was more than 6 months ago at time of delivery that had breakthrough covid infections: ' + str(df_breakthrough_two_shots_pyspark.count()))

df_breakthrough_boosted_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted_covid_breakthrough AS boosted WHERE boosted.covid_test_date >= '2021-12-25'")
print('# Number of women boosted at time of delivery that had breakthrough covid infections: ' + str(df_breakthrough_boosted_pyspark.count()))

df_breakthrough_vaccinated_but_not_boosted_matched_pyspark  = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_but_not_boosted_matched_control_covid AS matched WHERE matched.covid_test_date >= '2021-12-25'")
print('# Number of vaccinated, but not boosted matched pregnant women with a covid infection during pregnancy: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_pyspark.count()))


# evaluate conception dates of cohorts
df_breakthrough_jj_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_mrna_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_moderna_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_pfizer_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_one_mrna_pyspark.agg({'conception_date': 'min'}).show()
df_covid_immunity_pyspark.agg({'conception_date': 'min'}).show()
df_unvaccinated_pyspark.agg({'conception_date': 'min'}).show()
df_unvaccinated_matched_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_two_shots_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_boosted_pyspark.agg({'conception_date': 'min'}).show()
df_breakthrough_vaccinated_but_not_boosted_matched_pyspark.agg({'conception_date': 'min'}).show()


# add patient deaths by covid to table
df_unvaccinated_with_deaths = add_patient_deaths(df_unvaccinated_pyspark)
df_covid_immunity_with_deaths = add_patient_deaths(df_covid_immunity_pyspark)
df_breakthrough_mrna_with_deaths = add_patient_deaths(df_breakthrough_mrna_pyspark)
df_breakthrough_jj_with_deaths = add_patient_deaths(df_breakthrough_jj_pyspark)
df_breakthrough_moderna_with_deaths = add_patient_deaths(df_breakthrough_moderna_pyspark)
df_breakthrough_pfizer_with_deaths = add_patient_deaths(df_breakthrough_pfizer_pyspark)
df_breakthrough_one_mrna_with_deaths = add_patient_deaths(df_breakthrough_one_mrna_pyspark)
df_breakthrough_two_shots_with_deaths = add_patient_deaths(df_breakthrough_two_shots_pyspark)
df_breakthrough_boosted_with_deaths = add_patient_deaths(df_breakthrough_boosted_pyspark)
df_breakthrough_vaccinated_but_not_boosted_matched_with_deaths = add_patient_deaths(df_breakthrough_vaccinated_but_not_boosted_matched_pyspark)


# create diagnoses counts table
df_pat_conditions_covid_mom_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_covid_maternity_conditions").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_pat_conditions_covid_mom_pyspark.createOrReplaceTempView("diagnoses")

df_unvaccinated_with_deaths.createOrReplaceTempView("covid_mom")
df_unvaccinated_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


df_covid_immunity_with_deaths.createOrReplaceTempView("covid_mom")
df_covid_immunity_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


df_breakthrough_mrna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_mrna_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


df_breakthrough_jj_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_jj_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()

df_breakthrough_moderna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_moderna_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()

df_breakthrough_pfizer_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_pfizer_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()

df_breakthrough_one_mrna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_one_mrna_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


df_breakthrough_two_shots_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_two_shots_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


df_breakthrough_boosted_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_boosted_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()

df_breakthrough_vaccinated_but_not_boosted_matched_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_vaccinated_but_not_boosted_matched_diagnoses = spark.sql(
"""
FROM diagnoses AS d
INNER JOIN covid_mom AS cm
ON d.pat_id = cm.pat_id
  AND d.date_of_entry BETWEEN cm.covid_test_date AND cm.ob_delivery_delivery_date
SELECT d.pat_id, d.dx_id
""").toPandas()


# add medication counts to table
medication_count_unvaccinated = unique_medication_counts(df_unvaccinated_with_deaths)
medication_count_covid_immunity = unique_medication_counts(df_covid_immunity_with_deaths)
medication_count_breakthrough_mrna = unique_medication_counts(df_breakthrough_mrna_with_deaths)
medication_count_breakthrough_jj = unique_medication_counts(df_breakthrough_jj_with_deaths)
medication_count_breakthrough_moderna = unique_medication_counts(df_breakthrough_moderna_with_deaths)
medication_count_breakthrough_pfizer = unique_medication_counts(df_breakthrough_pfizer_with_deaths)
medication_count_breakthrough_one_mrna = unique_medication_counts(df_breakthrough_one_mrna_with_deaths)
medication_count_breakthrough_two_shots = unique_medication_counts(df_breakthrough_two_shots_with_deaths)
medication_count_breakthrough_boosted = unique_medication_counts(df_breakthrough_boosted_with_deaths)
medication_count_breakthrough_vaccinated_but_not_boosted_matched = unique_medication_counts(df_breakthrough_vaccinated_but_not_boosted_matched_with_deaths)


# add diagnoses counts to table
df_unvaccinated = add_diagnoses_counts(df_unvaccinated_diagnoses, df_unvaccinated_with_deaths.toPandas())
df_unvaccinated = df_unvaccinated.merge(medication_count_unvaccinated.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_unvaccinated = df_unvaccinated.fillna(0)

df_covid_immunity = add_diagnoses_counts(df_covid_immunity_diagnoses, df_covid_immunity_with_deaths.toPandas())
df_covid_immunity= df_covid_immunity.merge(medication_count_unvaccinated.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_covid_immunity = df_covid_immunity.fillna(0)

df_breakthrough_mrna = add_diagnoses_counts(df_breakthrough_mrna_diagnoses, df_breakthrough_mrna_with_deaths.toPandas())
df_breakthrough_mrna = df_breakthrough_mrna.merge(medication_count_breakthrough_mrna.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_mrna = df_breakthrough_mrna.fillna(0)

df_breakthrough_jj = add_diagnoses_counts(df_breakthrough_jj_diagnoses, df_breakthrough_jj_with_deaths.toPandas())
df_breakthrough_jj = df_breakthrough_jj.merge(medication_count_breakthrough_jj.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_jj = df_breakthrough_jj.fillna(0)

df_breakthrough_moderna = add_diagnoses_counts(df_breakthrough_moderna_diagnoses, df_breakthrough_moderna_with_deaths.toPandas())
df_breakthrough_moderna = df_breakthrough_moderna.merge(medication_count_breakthrough_moderna.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_moderna = df_breakthrough_moderna.fillna(0)

df_breakthrough_pfizer = add_diagnoses_counts(df_breakthrough_pfizer_diagnoses, df_breakthrough_pfizer_with_deaths.toPandas())
df_breakthrough_pfizer = df_breakthrough_pfizer.merge(medication_count_breakthrough_pfizer.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_pfizer = df_breakthrough_pfizer.fillna(0)

df_breakthrough_one_mrna = add_diagnoses_counts(df_breakthrough_one_mrna_diagnoses, df_breakthrough_one_mrna_with_deaths.toPandas())
df_breakthrough_one_mrna = df_breakthrough_one_mrna.merge(medication_count_breakthrough_one_mrna.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_one_mrna = df_breakthrough_one_mrna.fillna(0)

df_breakthrough_two_shots = add_diagnoses_counts(df_breakthrough_two_shots_diagnoses, df_breakthrough_two_shots_with_deaths.toPandas())
df_breakthrough_two_shots = df_breakthrough_two_shots.merge(medication_count_breakthrough_two_shots.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_two_shots = df_breakthrough_two_shots.fillna(0)

df_breakthrough_boosted = add_diagnoses_counts(df_breakthrough_boosted_diagnoses, df_breakthrough_boosted_with_deaths.toPandas())
df_breakthrough_boosted = df_breakthrough_boosted.merge(medication_count_breakthrough_boosted.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_boosted = df_breakthrough_boosted.fillna(0)

df_breakthrough_vaccinated_but_not_boosted_matched = add_diagnoses_counts(df_breakthrough_vaccinated_but_not_boosted_matched_diagnoses, df_breakthrough_vaccinated_but_not_boosted_matched_with_deaths.toPandas())
df_breakthrough_vaccinated_but_not_boosted_matched = df_breakthrough_vaccinated_but_not_boosted_matched.merge(medication_count_breakthrough_vaccinated_but_not_boosted_matched.toPandas(), left_on='pat_id', right_on='pat_id', how='left')
df_breakthrough_vaccinated_but_not_boosted_matched = df_breakthrough_vaccinated_but_not_boosted_matched.fillna(0)


# get covid encounter info
df_covid_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_adss_study_who_scores")
df_covid_pyspark.createOrReplaceTempView("covid")
df_unvaccinated_with_deaths.createOrReplaceTempView("covid_mom")
df_unvaccinated_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_mrna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_mrna_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_jj_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_jj_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_moderna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_moderna_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_pfizer_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_pfizer_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_one_mrna_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_one_mrna_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list,  c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_covid_immunity_with_deaths.createOrReplaceTempView("covid_mom")
df_covid_immunity_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list,  c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_two_shots_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_two_shots_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_boosted_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_boosted_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


df_breakthrough_vaccinated_but_not_boosted_matched_with_deaths.createOrReplaceTempView("covid_mom")
df_breakthrough_vaccinated_but_not_boosted_matched_covid_records = spark.sql(
"""
FROM covid AS c
INNER JOIN covid_mom AS cm
ON c.pat_id = cm.pat_id
  AND c.discharge_datetime_last > cm.covid_test_date
SELECT c.pat_id, c.who_score, c.oxygen_device_concat_ws, c.encounter_duration, c.bp_systolic_first, 
c.bp_diastolic_first, c.temperature_first, c.pulse_first, c.respirations_first,
c.base_patient_class_collect_list, c.patient_class_collect_list, c.vasopressor_sum
""").toPandas()


# format COVID encounter info
df_breakthrough_mrna_final = format_encounters(df_breakthrough_mrna_covid_records, df_breakthrough_mrna)
df_breakthrough_mrna_final = format_dataframe_for_modeling(df_breakthrough_mrna_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_jj_final = format_encounters(df_breakthrough_jj_covid_records, df_breakthrough_jj)
df_breakthrough_jj_final = format_dataframe_for_modeling(df_breakthrough_jj_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_moderna_final = format_encounters(df_breakthrough_moderna_covid_records, df_breakthrough_moderna)
df_breakthrough_moderna_final = format_dataframe_for_modeling(df_breakthrough_moderna_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_pfizer_final = format_encounters(df_breakthrough_pfizer_covid_records, df_breakthrough_pfizer)
df_breakthrough_pfizer_final = format_dataframe_for_modeling(df_breakthrough_pfizer_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_one_mrna_final = format_encounters(df_breakthrough_one_mrna_covid_records, df_breakthrough_one_mrna)
df_breakthrough_one_mrna_final = format_dataframe_for_modeling(df_breakthrough_one_mrna_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])
                                                                                                        
df_covid_immunity_final = format_encounters(df_covid_immunity_covid_records, df_covid_immunity)
df_covid_immunity_final = format_dataframe_for_modeling(df_covid_immunity_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_unvaccinated_final = format_encounters(df_unvaccinated_covid_records, df_unvaccinated)
df_unvaccinated_final = format_dataframe_for_modeling(df_unvaccinated_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])


# include 2 shots versus 3 shots analysis
df_breakthrough_two_shots_final = format_encounters(df_breakthrough_two_shots_covid_records, df_breakthrough_two_shots)
df_breakthrough_two_shots_final = format_dataframe_for_modeling(df_breakthrough_two_shots_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_boosted_final = format_encounters(df_breakthrough_boosted_covid_records, df_breakthrough_boosted)
df_breakthrough_boosted_final = format_dataframe_for_modeling(df_breakthrough_boosted_final).drop(columns=['pat_enc_csn_id_collect_list',	'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list'])

df_breakthrough_vaccinated_but_not_boosted_matched_final = format_encounters(df_breakthrough_vaccinated_but_not_boosted_matched_covid_records, df_breakthrough_vaccinated_but_not_boosted_matched)
df_breakthrough_vaccinated_but_not_boosted_matched_final = format_dataframe_for_modeling(df_breakthrough_vaccinated_but_not_boosted_matched_final)


# format unvaccinated matched cohort with covid encounter data
df_temp = df_unvaccinated_matched_pyspark.select('pat_id', 'instance', 'episode_id', 'child_episode_id').toPandas()
df_unvaccinated_matched_final = extract_matched_data_from_unmatched_df(df_temp, df_unvaccinated_final)
print("# Number of Unvaccinated Matched Pregnant People: " + str(len(df_unvaccinated_matched_final)))


# compare max oxygen device, oxygen assistance, and vasopressor use between cohorts
dict_unvaccinated_max_oxygen_device = make_oxygen_device_dict(df_unvaccinated_final['max_oxygen_device'])
dict_covid_immunity_max_oxygen_device = make_oxygen_device_dict(df_covid_immunity_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_covid_immunity_max_oxygen_device, dict_unvaccinated_max_oxygen_device)
calc_fishers_exact_test_not_simulated(make_oxygen_assistance_dict(dict_covid_immunity_max_oxygen_device), make_oxygen_assistance_dict(dict_unvaccinated_max_oxygen_device))

dict_unvaccinated_oxygen_assistance = make_oxygen_assistance_dict(dict_unvaccinated_max_oxygen_device)
conversion = dict_unvaccinated_oxygen_assistance['None']/sum(dict_unvaccinated_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_oxygen_assistance['Oxygen Assistance'], dict_unvaccinated_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_oxygen_assistance['Oxygen Assistance']/(dict_unvaccinated_oxygen_assistance['None'] + dict_unvaccinated_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_covid_immunity_oxygen_assistance = make_oxygen_assistance_dict(dict_covid_immunity_max_oxygen_device)
conversion = dict_covid_immunity_oxygen_assistance['None']/sum(dict_covid_immunity_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_covid_immunity_oxygen_assistance['Oxygen Assistance'], dict_covid_immunity_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_covid_immunity_oxygen_assistance['Oxygen Assistance']/(dict_covid_immunity_oxygen_assistance['None'] + dict_covid_immunity_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_covid_immunity_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_unvaccinated_vasopressor = make_vasopressor_dict(df_unvaccinated_final['vasopressor_sum'])
conversion = dict_unvaccinated_vasopressor['No']/sum(dict_unvaccinated_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_vasopressor['Yes'], dict_unvaccinated_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_vasopressor['Yes']/(dict_unvaccinated_vasopressor['No'] + dict_unvaccinated_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_covid_immunity_vasopressor = make_vasopressor_dict(df_covid_immunity_final['vasopressor_sum'])
conversion = dict_covid_immunity_vasopressor['No']/sum(dict_unvaccinated_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_covid_immunity_vasopressor['Yes'], dict_covid_immunity_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_covid_immunity_vasopressor['Yes']/(dict_covid_immunity_vasopressor['No'] + dict_covid_immunity_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_covid_immunity_vasopressor, dict_unvaccinated_vasopressor)

dict_breakthrough_mrna_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_mrna_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_max_oxygen_device, dict_unvaccinated_max_oxygen_device)

dict_vaccinated_mrna_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_mrna_max_oxygen_device)
conversion = dict_vaccinated_mrna_oxygen_assistance['None']/sum(dict_vaccinated_mrna_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_mrna_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_mrna_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_mrna_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_mrna_oxygen_assistance['None'] + dict_vaccinated_mrna_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_vaccinated_vasopressor = make_vasopressor_dict(df_breakthrough_mrna_final['vasopressor_sum'])
conversion = dict_vaccinated_vasopressor['No']/sum(dict_vaccinated_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_vasopressor['Yes'], dict_vaccinated_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_vasopressor['Yes']/(dict_vaccinated_vasopressor['No'] + dict_vaccinated_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_vasopressor, dict_unvaccinated_vasopressor)

dict_unvaccinated_matched_max_oxygen_device = make_oxygen_device_dict(df_unvaccinated_matched_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_max_oxygen_device, dict_unvaccinated_matched_max_oxygen_device)

dict_unvaccinated_matched_oxygen_assistance = make_oxygen_assistance_dict(dict_unvaccinated_matched_max_oxygen_device)
conversion = dict_unvaccinated_matched_oxygen_assistance['None']/sum(dict_unvaccinated_matched_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_matched_oxygen_assistance['Oxygen Assistance'], dict_unvaccinated_matched_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_matched_oxygen_assistance['Oxygen Assistance']/(dict_unvaccinated_matched_oxygen_assistance['None'] + dict_unvaccinated_matched_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_oxygen_assistance, dict_unvaccinated_matched_oxygen_assistance)

dict_unvaccinated_matched_vasopressor = make_vasopressor_dict(df_unvaccinated_matched_final['vasopressor_sum'])
conversion = dict_unvaccinated_matched_vasopressor['No']/sum(dict_unvaccinated_matched_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_matched_vasopressor['Yes'], dict_unvaccinated_matched_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_matched_vasopressor['Yes']/(dict_unvaccinated_matched_vasopressor['No'] + dict_unvaccinated_matched_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_vasopressor, dict_unvaccinated_matched_vasopressor)

dict_breakthrough_jj_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_jj_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_jj_max_oxygen_device, dict_unvaccinated_max_oxygen_device)

dict_vaccinated_jj_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_jj_max_oxygen_device)
conversion = dict_vaccinated_jj_oxygen_assistance['None']/sum(dict_vaccinated_jj_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_jj_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_jj_oxygen_assistance['None'] + dict_vaccinated_jj_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_vaccinated_jj_vasopressor = make_vasopressor_dict(df_breakthrough_jj_final['vasopressor_sum'])
conversion = dict_vaccinated_jj_vasopressor['No']/sum(dict_vaccinated_jj_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_vasopressor['Yes'], dict_vaccinated_jj_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_vasopressor['Yes']/(dict_vaccinated_jj_vasopressor['No'] + dict_vaccinated_jj_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_vasopressor, dict_unvaccinated_vasopressor)

dict_breakthrough_one_mrna_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_one_mrna_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_one_mrna_max_oxygen_device, dict_unvaccinated_max_oxygen_device)

dict_vaccinated_one_mrna_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_one_mrna_max_oxygen_device)
conversion = dict_vaccinated_one_mrna_oxygen_assistance['None']/sum(dict_vaccinated_one_mrna_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_one_mrna_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_one_mrna_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_one_mrna_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_one_mrna_oxygen_assistance['None'] + dict_vaccinated_one_mrna_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_one_mrna_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_vaccinated_one_mrna_vasopressor = make_vasopressor_dict(df_breakthrough_one_mrna_final['vasopressor_sum'])
conversion = dict_vaccinated_one_mrna_vasopressor['No']/sum(dict_vaccinated_one_mrna_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_one_mrna_vasopressor['Yes'], dict_vaccinated_one_mrna_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_one_mrna_vasopressor['Yes']/(dict_vaccinated_one_mrna_vasopressor['No'] + dict_vaccinated_one_mrna_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_one_mrna_vasopressor, dict_unvaccinated_vasopressor)

dict_breakthrough_moderna_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_moderna_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_moderna_max_oxygen_device, dict_unvaccinated_max_oxygen_device)

dict_vaccinated_moderna_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_moderna_max_oxygen_device)
conversion = dict_vaccinated_moderna_oxygen_assistance['None']/sum(dict_vaccinated_moderna_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_moderna_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_moderna_oxygen_assistance['None'] + dict_vaccinated_moderna_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_vaccinated_moderna_vasopressor = make_vasopressor_dict(df_breakthrough_moderna_final['vasopressor_sum'])
conversion = dict_vaccinated_moderna_vasopressor['No']/sum(dict_vaccinated_moderna_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_vasopressor['Yes'], dict_vaccinated_moderna_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_vasopressor['Yes']/(dict_vaccinated_moderna_vasopressor['No'] + dict_vaccinated_moderna_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_vasopressor, dict_unvaccinated_vasopressor)

dict_breakthrough_pfizer_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_pfizer_final['max_oxygen_device'])
calc_fishers_exact_test_not_simulated(dict_breakthrough_pfizer_max_oxygen_device, dict_unvaccinated_max_oxygen_device)

dict_vaccinated_pfizer_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_pfizer_max_oxygen_device)
conversion = dict_vaccinated_pfizer_oxygen_assistance['None']/sum(dict_vaccinated_pfizer_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_pfizer_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_pfizer_oxygen_assistance['None'] + dict_vaccinated_pfizer_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_oxygen_assistance, dict_unvaccinated_oxygen_assistance)

dict_vaccinated_pfizer_vasopressor = make_vasopressor_dict(df_breakthrough_pfizer_final['vasopressor_sum'])
conversion = dict_vaccinated_pfizer_vasopressor['No']/sum(dict_vaccinated_pfizer_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_vasopressor['Yes'], dict_vaccinated_pfizer_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_vasopressor['Yes']/(dict_vaccinated_pfizer_vasopressor['No'] + dict_vaccinated_pfizer_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_vasopressor, dict_unvaccinated_vasopressor)

dict_breakthrough_two_shots_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_two_shots_final['max_oxygen_device'])
dict_vaccinated_two_shots_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_two_shots_max_oxygen_device)
conversion = dict_vaccinated_two_shots_oxygen_assistance['None']/sum(dict_vaccinated_two_shots_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_two_shots_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_two_shots_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_two_shots_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_two_shots_oxygen_assistance['None'] + dict_vaccinated_two_shots_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_breakthrough_boosted_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_boosted_final['max_oxygen_device'])
dict_vaccinated_boosted_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_boosted_max_oxygen_device)
conversion = dict_vaccinated_boosted_oxygen_assistance['None']/sum(dict_vaccinated_boosted_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_boosted_oxygen_assistance['Oxygen Assistance'], dict_vaccinated_boosted_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_boosted_oxygen_assistance['Oxygen Assistance']/(dict_vaccinated_boosted_oxygen_assistance['None'] + dict_vaccinated_boosted_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


calc_fishers_exact_test_not_simulated(dict_breakthrough_two_shots_max_oxygen_device, dict_breakthrough_boosted_max_oxygen_device)
calc_fishers_exact_test_not_simulated(dict_vaccinated_two_shots_oxygen_assistance, dict_vaccinated_boosted_oxygen_assistance)


dict_vaccinated_two_shots_vasopressor = make_vasopressor_dict(df_breakthrough_two_shots_final['vasopressor_sum'])
conversion = dict_vaccinated_two_shots_vasopressor['No']/sum(dict_vaccinated_two_shots_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_two_shots_vasopressor['Yes'], dict_vaccinated_two_shots_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_two_shots_vasopressor['Yes']/(dict_vaccinated_two_shots_vasopressor['No'] + dict_vaccinated_jj_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_vaccinated_boosted_vasopressor = make_vasopressor_dict(df_breakthrough_boosted_final['vasopressor_sum'])
conversion = dict_vaccinated_boosted_vasopressor['No']/sum(dict_vaccinated_boosted_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_boosted_vasopressor['Yes'], dict_vaccinated_boosted_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_boosted_vasopressor['Yes']/(dict_vaccinated_boosted_vasopressor['No'] + dict_vaccinated_boosted_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_two_shots_vasopressor, dict_vaccinated_boosted_vasopressor)

dict_breakthrough_vaccinated_but_not_boosted_matched_max_oxygen_device = make_oxygen_device_dict(df_breakthrough_vaccinated_but_not_boosted_matched_final['max_oxygen_device'])
dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance = make_oxygen_assistance_dict(dict_breakthrough_vaccinated_but_not_boosted_matched_max_oxygen_device)
conversion = dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['None']/sum(dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance.values())
ci_low, ci_up = proportion_confint(dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['Oxygen Assistance'], dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['None'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['Oxygen Assistance']/(dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['None'] + dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance['Oxygen Assistance']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_max_oxygen_device, dict_breakthrough_boosted_max_oxygen_device)
calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_oxygen_assistance, dict_vaccinated_boosted_oxygen_assistance)


dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor = make_vasopressor_dict(df_breakthrough_vaccinated_but_not_boosted_matched_final['vasopressor_sum'])
conversion = dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['No']/sum(dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor.values())
ci_low, ci_up = proportion_confint(dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['Yes'], dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['Yes']/(dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['No'] + dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_vasopressor, dict_vaccinated_boosted_vasopressor)


# evaluate covid-19 severity according to the WHO COVID-19 Severity scale
dict_unvaccinated_who_score = make_covid_severity_dict(df_unvaccinated_final)
dict_covid_immunity_who_score = make_covid_severity_dict(df_covid_immunity_final)
calc_fishers_exact_test_not_simulated(dict_covid_immunity_who_score, dict_unvaccinated_who_score)

dict_breakthrough_mrna_who_score = make_covid_severity_dict(df_breakthrough_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_who_score, dict_unvaccinated_who_score)

dict_unvaccinated_matched_who_score = make_covid_severity_dict(df_unvaccinated_matched_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_who_score, dict_unvaccinated_matched_who_score)

dict_breakthrough_jj_who_score = make_covid_severity_dict(df_breakthrough_jj_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_jj_who_score, dict_unvaccinated_who_score)

dict_breakthrough_one_mrna_who_score = make_covid_severity_dict(df_breakthrough_one_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_one_mrna_who_score, dict_unvaccinated_who_score)

dict_breakthrough_moderna_who_score = make_covid_severity_dict(df_breakthrough_moderna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_moderna_who_score, dict_unvaccinated_who_score)

dict_breakthrough_pfizer_who_score = make_covid_severity_dict(df_breakthrough_pfizer_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_pfizer_who_score, dict_unvaccinated_who_score)

dict_breakthrough_two_shots_who_score = make_covid_severity_dict(df_breakthrough_two_shots_final)
dict_breakthrough_boosted_who_score = make_covid_severity_dict(df_breakthrough_boosted_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_two_shots_who_score, dict_breakthrough_boosted_who_score)

dict_breakthrough_vaccinated_but_not_boosted_matched_who_score = make_covid_severity_dict(df_breakthrough_vaccinated_but_not_boosted_matched_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_who_score, dict_breakthrough_boosted_who_score)


# compare max patient class, hospitalization rates, and inpatient rates between cohorts
dict_unvaccinated_patient_class = make_patient_class_dict(df_unvaccinated_final)
dict_covid_immunity_patient_class = make_patient_class_dict(df_covid_immunity_final)
calc_fishers_exact_test_not_simulated(dict_covid_immunity_patient_class, dict_unvaccinated_patient_class)
print('\n')

dict_unvaccinated_inpatient = make_inpatient_dict(df_unvaccinated_final)
dict_covid_immunity_inpatient = make_inpatient_dict(df_covid_immunity_final)
calc_fishers_exact_test_not_simulated(dict_covid_immunity_inpatient, dict_unvaccinated_inpatient)
print('\n')

dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
dict_covid_immunity_hospitalization = make_hospitalization_dict(df_covid_immunity_final)
calc_fishers_exact_test_not_simulated(dict_covid_immunity_hospitalization, dict_unvaccinated_hospitalization)

dict_unvaccinated_inpatient = determine_inpatient_status(df_unvaccinated_final)
conversion = dict_unvaccinated_inpatient['No']/sum(dict_unvaccinated_inpatient.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_inpatient['Yes'], dict_unvaccinated_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_inpatient['Yes']/(dict_unvaccinated_inpatient['No'] + dict_unvaccinated_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_covid_immunity_inpatient = determine_inpatient_status(df_covid_immunity_final)
conversion = dict_covid_immunity_inpatient['No']/sum(dict_covid_immunity_inpatient.values())
ci_low, ci_up = proportion_confint(dict_covid_immunity_inpatient['Yes'], dict_covid_immunity_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_covid_immunity_inpatient['Yes']/(dict_covid_immunity_inpatient['No'] + dict_covid_immunity_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_covid_immunity_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_inpatient = determine_inpatient_status(df_unvaccinated_final)
conversion = dict_unvaccinated_inpatient['No']/sum(dict_unvaccinated_inpatient.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_inpatient['Yes'], dict_unvaccinated_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_inpatient['Yes']/(dict_unvaccinated_inpatient['No'] + dict_unvaccinated_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_covid_immunity_inpatient = determine_inpatient_status(df_covid_immunity_final)
conversion = dict_covid_immunity_inpatient['No']/sum(dict_covid_immunity_inpatient.values())
ci_low, ci_up = proportion_confint(dict_covid_immunity_inpatient['Yes'], dict_covid_immunity_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_covid_immunity_inpatient['Yes']/(dict_covid_immunity_inpatient['No'] + dict_covid_immunity_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_covid_immunity_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
dict_covid_induced_immunity_hospitalization = determine_hospitalization_status(df_covid_immunity_final)
conversion = dict_covid_induced_immunity_hospitalization['No']/sum(dict_covid_induced_immunity_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_hospitalization['Yes'], dict_unvaccinated_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_covid_induced_immunity_hospitalization['Yes']/(dict_covid_induced_immunity_hospitalization['No'] + dict_covid_induced_immunity_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_covid_induced_immunity_hospitalization, dict_unvaccinated_hospitalization))

dict_breakthrough_mrna_patient_class = make_patient_class_dict(df_breakthrough_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_patient_class, dict_unvaccinated_patient_class)
print('\n')

dict_breakthrough_mrna_inpatient = determine_inpatient_status(df_breakthrough_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_inpatient, dict_unvaccinated_inpatient)
print('\n')

dict_breakthrough_mrna_hospitalization = make_hospitalization_dict(df_breakthrough_mrna_final)
dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_hospitalization, dict_unvaccinated_hospitalization)

dict_vaccinated_inpatient = determine_inpatient_status(df_breakthrough_mrna_final)
conversion = dict_vaccinated_inpatient['No']/sum(dict_vaccinated_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_inpatient['Yes'], dict_vaccinated_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_inpatient['Yes']/(dict_vaccinated_inpatient['No'] + dict_vaccinated_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
conversion = dict_unvaccinated_hospitalization['No']/sum(dict_unvaccinated_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_hospitalization['Yes'], dict_unvaccinated_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_hospitalization['Yes']/(dict_unvaccinated_hospitalization['No'] + dict_unvaccinated_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_vaccinated_hospitalization = determine_hospitalization_status(df_breakthrough_mrna_final)
conversion = dict_vaccinated_hospitalization['No']/sum(dict_vaccinated_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_hospitalization['Yes'], dict_vaccinated_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_hospitalization['Yes']/(dict_vaccinated_hospitalization['No'] + dict_vaccinated_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_vaccinated_hospitalization, dict_unvaccinated_hospitalization))

dict_unvaccinated_matched_patient_class = make_patient_class_dict(df_unvaccinated_matched_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_patient_class, dict_unvaccinated_matched_patient_class)
print('\n')

dict_unvaccinated_matched_inpatient = make_inpatient_dict(df_unvaccinated_matched_final)
dict_breakthrough_mrna_inpatient = make_inpatient_dict(df_breakthrough_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_inpatient, dict_unvaccinated_matched_inpatient)
print('\n')

dict_unvaccinated_matched_hospitalization = make_hospitalization_dict(df_unvaccinated_matched_final)
dict_breakthrough_mrna_hospitalization = make_hospitalization_dict(df_breakthrough_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_mrna_hospitalization, dict_unvaccinated_matched_hospitalization)

dict_vaccinated_inpatient = determine_inpatient_status(df_breakthrough_mrna_final)
dict_unvaccinated_matched_inpatient = determine_inpatient_status(df_unvaccinated_matched_final)
conversion = dict_unvaccinated_matched_inpatient['No']/sum(dict_unvaccinated_matched_inpatient.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_matched_inpatient['Yes'], dict_unvaccinated_matched_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_matched_inpatient['Yes']/(dict_unvaccinated_matched_inpatient['No'] + dict_unvaccinated_matched_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_inpatient, dict_unvaccinated_matched_inpatient))

dict_vaccinated_hospitalization = determine_hospitalization_status(df_breakthrough_mrna_final)
conversion = dict_vaccinated_hospitalization['No']/sum(dict_vaccinated_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_hospitalization['Yes'], dict_vaccinated_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_hospitalization['Yes']/(dict_vaccinated_hospitalization['No'] + dict_vaccinated_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_unvaccinated_matched_hospitalization = determine_hospitalization_status(df_unvaccinated_matched_final)
conversion = dict_unvaccinated_matched_hospitalization['No']/sum(dict_unvaccinated_matched_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_matched_hospitalization['Yes'], dict_unvaccinated_matched_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_matched_hospitalization['Yes']/(dict_unvaccinated_matched_hospitalization['No'] + dict_unvaccinated_matched_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_vaccinated_hospitalization, dict_unvaccinated_matched_hospitalization))

dict_breakthrough_jj_patient_class = make_patient_class_dict(df_breakthrough_jj_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_jj_patient_class, dict_unvaccinated_patient_class)

print('\n')

dict_breakthrough_jj_inpatient = determine_inpatient_status(df_breakthrough_jj_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_jj_inpatient, dict_unvaccinated_inpatient)

print('\n')

dict_breakthrough_jj_hospitalization = make_hospitalization_dict(df_breakthrough_jj_final)
dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_jj_hospitalization, dict_unvaccinated_hospitalization)

dict_vaccinated_jj_inpatient = determine_inpatient_status(df_breakthrough_jj_final)
conversion = dict_vaccinated_jj_inpatient['No']/sum(dict_vaccinated_jj_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_inpatient['Yes'], dict_vaccinated_jj_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_inpatient['Yes']/(dict_vaccinated_jj_inpatient['No'] + dict_vaccinated_jj_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
dict_vaccinated_jj_hospitalization = determine_hospitalization_status(df_breakthrough_jj_final)
conversion = dict_vaccinated_jj_hospitalization['No']/sum(dict_vaccinated_jj_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_hospitalization['Yes'], dict_vaccinated_jj_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_hospitalization['Yes']/(dict_vaccinated_jj_hospitalization['No'] + dict_vaccinated_jj_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_hospitalization, dict_unvaccinated_hospitalization))

dict_breakthrough_one_mrna_patient_class = make_patient_class_dict(df_breakthrough_one_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_one_mrna_patient_class, dict_unvaccinated_patient_class)

print('\n')

dict_breakthrough_one_mrna_inpatient = determine_inpatient_status(df_breakthrough_one_mrna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_one_mrna_inpatient, dict_unvaccinated_inpatient)

print('\n')

dict_breakthrough_one_mrna_hospitalization = make_hospitalization_dict(df_breakthrough_one_mrna_final)
dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_one_mrna_hospitalization, dict_unvaccinated_hospitalization)

dict_vaccinated_one_mrna_inpatient = determine_inpatient_status(df_breakthrough_one_mrna_final)
conversion = dict_vaccinated_one_mrna_inpatient['No']/sum(dict_vaccinated_one_mrna_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_one_mrna_inpatient['Yes'], dict_vaccinated_one_mrna_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_one_mrna_inpatient['Yes']/(dict_vaccinated_one_mrna_inpatient['No'] + dict_vaccinated_one_mrna_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_one_mrna_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
dict_vaccinated_one_mrna_hospitalization = determine_hospitalization_status(df_breakthrough_one_mrna_final)
conversion = dict_vaccinated_one_mrna_hospitalization['No']/sum(dict_vaccinated_one_mrna_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_one_mrna_hospitalization['Yes'], dict_vaccinated_one_mrna_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_one_mrna_hospitalization['Yes']/(dict_vaccinated_one_mrna_hospitalization['No'] + dict_vaccinated_one_mrna_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_one_mrna_hospitalization, dict_unvaccinated_hospitalization))

dict_breakthrough_moderna_patient_class = make_patient_class_dict(df_breakthrough_moderna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_moderna_patient_class, dict_unvaccinated_patient_class)
print('\n')

dict_breakthrough_moderna_inpatient = determine_inpatient_status(df_breakthrough_moderna_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_moderna_inpatient, dict_unvaccinated_inpatient)

print('\n')

dict_breakthrough_moderna_hospitalization = make_hospitalization_dict(df_breakthrough_moderna_final)
dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_moderna_hospitalization, dict_unvaccinated_hospitalization)

dict_vaccinated_moderna_inpatient = determine_inpatient_status(df_breakthrough_moderna_final)
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_inpatient['Yes'], dict_vaccinated_moderna_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_inpatient['Yes']/(dict_vaccinated_moderna_inpatient['No'] + dict_vaccinated_moderna_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
dict_vaccinated_moderna_hospitalization = determine_hospitalization_status(df_breakthrough_moderna_final)
conversion = dict_vaccinated_moderna_hospitalization['No']/sum(dict_vaccinated_moderna_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_hospitalization['Yes'], dict_vaccinated_moderna_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_hospitalization['Yes']/(dict_vaccinated_moderna_hospitalization['No'] + dict_vaccinated_moderna_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_hospitalization, dict_unvaccinated_hospitalization))

dict_breakthrough_pfizer_patient_class = make_patient_class_dict(df_breakthrough_pfizer_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_pfizer_patient_class, dict_unvaccinated_patient_class)
print('\n')

dict_breakthrough_pfizer_inpatient = determine_inpatient_status(df_breakthrough_pfizer_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_pfizer_inpatient, dict_unvaccinated_inpatient)
print('\n')

dict_breakthrough_pfizer_hospitalization = make_hospitalization_dict(df_breakthrough_pfizer_final)
dict_unvaccinated_hospitalization = make_hospitalization_dict(df_unvaccinated_final)
calc_fishers_exact_test_not_simulated(dict_breakthrough_pfizer_hospitalization, dict_unvaccinated_hospitalization)

dict_vaccinated_pfizer_inpatient = determine_inpatient_status(df_breakthrough_pfizer_final)
conversion = dict_vaccinated_pfizer_inpatient['No']/sum(dict_vaccinated_pfizer_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_inpatient['Yes'], dict_vaccinated_pfizer_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_inpatient['Yes']/(dict_vaccinated_pfizer_inpatient['No'] + dict_vaccinated_pfizer_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_inpatient, dict_unvaccinated_inpatient))

dict_unvaccinated_hospitalization = determine_hospitalization_status(df_unvaccinated_final)
dict_vaccinated_pfizer_hospitalization = determine_hospitalization_status(df_breakthrough_pfizer_final)
conversion = dict_vaccinated_pfizer_hospitalization['No']/sum(dict_vaccinated_pfizer_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_hospitalization['Yes'], dict_vaccinated_pfizer_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_hospitalization['Yes']/(dict_vaccinated_pfizer_hospitalization['No'] + dict_vaccinated_pfizer_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_hospitalization, dict_unvaccinated_hospitalization))

dict_vaccinated_moderna_hospitalization = determine_hospitalization_status(df_breakthrough_moderna_final)
conversion = dict_vaccinated_moderna_hospitalization['No']/sum(dict_vaccinated_moderna_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_hospitalization['Yes'], dict_vaccinated_moderna_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_hospitalization['Yes']/(dict_vaccinated_moderna_hospitalization['No'] + dict_vaccinated_moderna_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_vaccinated_pfizer_hospitalization = determine_hospitalization_status(df_breakthrough_pfizer_final)
conversion = dict_vaccinated_pfizer_hospitalization['No']/sum(dict_vaccinated_pfizer_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_hospitalization['Yes'], dict_vaccinated_pfizer_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_hospitalization['Yes']/(dict_vaccinated_pfizer_hospitalization['No'] + dict_vaccinated_pfizer_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_hospitalization, dict_vaccinated_pfizer_hospitalization))


dict_breakthrough_two_shots_patient_class = make_patient_class_dict(df_breakthrough_two_shots_final)
dict_breakthrough_two_shots_inpatient = determine_inpatient_status(df_breakthrough_two_shots_final)
dict_breakthrough_two_shots_hospitalization = make_hospitalization_dict(df_breakthrough_two_shots_final)
print('\n')


dict_breakthrough_boosted_patient_class = make_patient_class_dict(df_breakthrough_boosted_final)
dict_breakthrough_boosted_inpatient = determine_inpatient_status(df_breakthrough_boosted_final)
dict_breakthrough_boosted_hospitalization = make_hospitalization_dict(df_breakthrough_boosted_final)
print('\n')


calc_fishers_exact_test_not_simulated(dict_breakthrough_two_shots_patient_class, dict_breakthrough_boosted_patient_class)
print('\n')
calc_fishers_exact_test_not_simulated(dict_breakthrough_two_shots_inpatient, dict_breakthrough_boosted_inpatient)
print('\n')
calc_fishers_exact_test_not_simulated(dict_breakthrough_two_shots_hospitalization, dict_breakthrough_boosted_hospitalization)

dict_vaccinated_two_shots_inpatient = determine_inpatient_status(df_breakthrough_two_shots_final)
conversion = dict_vaccinated_two_shots_inpatient['No']/sum(dict_vaccinated_two_shots_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_two_shots_inpatient['Yes'], dict_vaccinated_two_shots_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_two_shots_inpatient['Yes']/(dict_vaccinated_pfizer_inpatient['No'] + dict_vaccinated_two_shots_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_vaccinated_boosted_inpatient = determine_inpatient_status(df_breakthrough_boosted_final)
conversion = dict_vaccinated_boosted_inpatient['No']/sum(dict_vaccinated_boosted_inpatient.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_boosted_inpatient['Yes'], dict_vaccinated_boosted_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_boosted_inpatient['Yes']/(dict_vaccinated_boosted_inpatient['No'] + dict_vaccinated_boosted_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_vaccinated_two_shots_inpatient, dict_vaccinated_boosted_inpatient))


dict_vaccinated_two_shots_hospitalization = determine_hospitalization_status(df_breakthrough_two_shots_final)
conversion = dict_vaccinated_two_shots_hospitalization['No']/sum(dict_vaccinated_two_shots_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_two_shots_hospitalization['Yes'], dict_vaccinated_two_shots_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_two_shots_hospitalization['Yes']/(dict_vaccinated_two_shots_hospitalization['No'] + dict_vaccinated_two_shots_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


dict_vaccinated_boosted_hospitalization = determine_hospitalization_status(df_breakthrough_boosted_final)
conversion = dict_vaccinated_boosted_hospitalization['No']/sum(dict_vaccinated_boosted_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_boosted_hospitalization['Yes'], dict_vaccinated_boosted_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_boosted_hospitalization['Yes']/(dict_vaccinated_boosted_hospitalization['No'] + dict_vaccinated_boosted_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_vaccinated_two_shots_hospitalization, dict_vaccinated_boosted_hospitalization))

dict_breakthrough_vaccinated_but_not_boosted_matched_patient_class = make_patient_class_dict(df_breakthrough_vaccinated_but_not_boosted_matched_final)
dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient = determine_inpatient_status(df_breakthrough_vaccinated_but_not_boosted_matched_final)
dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization = make_hospitalization_dict(df_breakthrough_vaccinated_but_not_boosted_matched_final)
print('\n')


calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_patient_class, dict_breakthrough_boosted_patient_class)
print('\n')
calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient, dict_breakthrough_boosted_inpatient)
print('\n')
calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization, dict_breakthrough_boosted_hospitalization)

dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient = determine_inpatient_status(df_breakthrough_vaccinated_but_not_boosted_matched_final)
conversion = dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['No']/sum(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient.values())
ci_low, ci_up = proportion_confint(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['Yes'], dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['Yes']/(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['No'] + dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_inpatient, dict_vaccinated_boosted_inpatient))

dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization = determine_hospitalization_status(df_breakthrough_vaccinated_but_not_boosted_matched_final)
conversion = dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['No']/sum(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization.values())
ci_low, ci_up = proportion_confint(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['Yes'], dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['Yes']/(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['No'] + dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization['Yes']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')


print(calc_fishers_exact_test_not_simulated(dict_breakthrough_vaccinated_but_not_boosted_matched_hospitalization, dict_vaccinated_boosted_hospitalization))


# evaluate medications and diagnoses during encounter for each cohort
print('# Number of Inpatients Unvaccinated: ' + str(len(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Max Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Median Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('\n')

print('# Min Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final.unique_outpatient_medication_count)))
print('\n')

print('# Min Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final.unique_medication_count)))
print('\n')

print('# Min Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('\n')

print('# Min Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Unvaccinated Inpatient: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses)))


print('# Number of Inpatients Unvaccinated: ' + str(len(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Max Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Median Number of Unique Inpatient Medications Unvaccinated: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('\n')

print('# Min Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final.unique_outpatient_medication_count)))
print('\n')

print('# Min Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Unvaccinated: ' + str(df_unvaccinated_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Unvaccinated: ' + str(stats.iqr(df_unvaccinated_final.unique_medication_count)))
print('\n')

print('# Min Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Unvaccinated Inpatient: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('\n')

print('# Min Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Unvaccinated Inpatient: ' + str(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Unvaccinated Inpatient: ' + str(stats.iqr(df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1].n_total_diagnoses)))


print('# Number of Inpatients Unvaccinated Matched: ' + str(len(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Max Number of Unique Inpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Median Number of Unique Inpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Unvaccinated Matched: ' + str(stats.iqr(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Vaccinated (mRNA) vs Unvaccinated Matched Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_mrna_final[df_breakthrough_mrna_final['max_patient_class'] > 1], df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Unvaccinated Matched: ' + str(stats.iqr(df_unvaccinated_matched_final.unique_outpatient_medication_count)))
print('# Vaccinated (mRNA) vs Unvaccinated Matched Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_mrna_final, df_unvaccinated_matched_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Unvaccinated Matched: ' + str(df_unvaccinated_matched_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Unvaccinated Matched: ' + str(stats.iqr(df_unvaccinated_matched_final.unique_medication_count)))
print('# Vaccinated (mRNA) vs Unvaccinated Matched Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_mrna_final, df_unvaccinated_matched_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Unvaccinated Matched Inpatient: ' + str(stats.iqr(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated (mRNA) vs Unvaccinated Matched Unique Diagnoses Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_mrna_final[df_breakthrough_mrna_final['max_patient_class'] > 1], df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Unvaccinated Matched Inpatient: ' + str(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Unvaccinated Matched Inpatient: ' + str(stats.iqr(df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated (mRNA) vs Unvaccinated Matched Total Diagnoses Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_mrna_final[df_breakthrough_mrna_final['max_patient_class'] > 1], df_unvaccinated_matched_final[df_unvaccinated_matched_final['max_patient_class'] > 1], 


	print('# Number of Inpatients Vaccinated (jj): ' + str(len(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (jj): ' + str(stats.iqr(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Vaccinated (jj) vs Unvaccinated Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (jj): ' + str(stats.iqr(df_breakthrough_jj_final.unique_outpatient_medication_count)))
print('# Vaccinated (jj) vs Unvaccinated Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_jj_final, df_unvaccinated_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (jj): ' + str(df_breakthrough_jj_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (jj): ' + str(stats.iqr(df_breakthrough_jj_final.unique_medication_count)))
print('# Vaccinated (jj) vs Unvaccinated Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_jj_final, df_unvaccinated_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (jj) Inpatient: ' + str(stats.iqr(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated (jj) vs Diagnoses Unique Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (jj) Inpatient: ' + str(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (jj) Inpatient: ' + str(stats.iqr(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated (jj) vs Diagnoses Total Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_jj_final[df_breakthrough_jj_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


print('# Number of Inpatients Vaccinated (one mrna): ' + str(len(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (one mrna): ' + str(stats.iqr(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Vaccinated (one mrna) vs Unvaccinated Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (one mrna): ' + str(stats.iqr(df_breakthrough_one_mrna_final.unique_outpatient_medication_count)))
print('# Vaccinated (one mrna) vs Unvaccinated Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_one_mrna_final, df_unvaccinated_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (one mrna): ' + str(df_breakthrough_one_mrna_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (one mrna): ' + str(stats.iqr(df_breakthrough_one_mrna_final.unique_medication_count)))
print('# Vaccinated (one mrna) vs Unvaccinated Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_one_mrna_final, df_unvaccinated_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (one mrna) Inpatient: ' + str(stats.iqr(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated (one mrna) vs Diagnoses Unique Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (one mrna) Inpatient: ' + str(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (one mrna) Inpatient: ' + str(stats.iqr(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated (one mrna) vs Diagnoses Total Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_one_mrna_final[df_breakthrough_one_mrna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


print('# Number of Inpatients Vaccinated (moderna): ' + str(len(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (moderna): ' + str(stats.iqr(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Vaccinated (moderna) vs Unvaccinated Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (moderna): ' + str(stats.iqr(df_breakthrough_moderna_final.unique_outpatient_medication_count)))
print('# Vaccinated (moderna) vs Unvaccinated Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_moderna_final, df_unvaccinated_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (moderna): ' + str(df_breakthrough_moderna_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (moderna): ' + str(stats.iqr(df_breakthrough_moderna_final.unique_medication_count)))
print('# Vaccinated (moderna) vs Unvaccinated Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_moderna_final, df_unvaccinated_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (moderna) Inpatient: ' + str(stats.iqr(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated (moderna) vs Unvaccinated Unique Diagnoses Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (moderna) Inpatient: ' + str(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (moderna) Inpatient: ' + str(stats.iqr(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated (moderna) vs Unvaccinated Total Diagnoses Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_moderna_final[df_breakthrough_moderna_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


print('# Number of Inpatients Vaccinated (pfizer): ' + str(len(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (pfizer): ' + str(stats.iqr(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Vaccinated (pfizer) vs Unvaccinated Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (pfizer): ' + str(stats.iqr(df_breakthrough_pfizer_final.unique_outpatient_medication_count)))
print('# Vaccinated (pfizer) vs Unvaccinated Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_pfizer_final, df_unvaccinated_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (pfizer): ' + str(df_breakthrough_pfizer_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (pfizer): ' + str(stats.iqr(df_breakthrough_pfizer_final.unique_medication_count)))
print('# Vaccinated (pfizer) vs Unvaccinated Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_pfizer_final, df_unvaccinated_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (pfizer) Inpatients: ' + str(stats.iqr(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated (pfizer) vs Unvaccinated Unique Diagnoses Count Inpatients p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (pfizer) Inpatients: ' + str(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (pfizer) Inpatients: ' + str(stats.iqr(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated (pfizer) vs Unvaccinated Total Diagnoses Count Inpatients p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_pfizer_final[df_breakthrough_pfizer_final['max_patient_class'] > 1], df_unvaccinated_final[df_unvaccinated_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


print('# Number of Inpatients Vaccinated (Two Shots): ' + str(len(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (Two Shots): ' + str(stats.iqr(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (Two Shots): ' + str(stats.iqr(df_breakthrough_two_shots_final.unique_outpatient_medication_count)))
print('\n')

print('# Min Number of Unique Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (Two Shots): ' + str(df_breakthrough_two_shots_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (Two Shots): ' + str(stats.iqr(df_breakthrough_two_shots_final.unique_medication_count)))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(stats.iqr(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (Two Shots) Inpatient: ' + str(stats.iqr(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1].n_total_diagnoses)))


print('# Number of Inpatients Vaccinated (Boosted): ' + str(len(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated (jBoosted): ' + str(stats.iqr(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('# Two Shots vs Boosted Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated (Boosted): ' + str(stats.iqr(df_breakthrough_boosted_final.unique_outpatient_medication_count)))
print('# Two Shots vs Boosted Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_two_shots_final, df_breakthrough_boosted_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated (Boosted): ' + str(df_breakthrough_boosted_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated (Boosted): ' + str(stats.iqr(df_breakthrough_boosted_final.unique_medication_count)))
print('# Two Shots vs Boosted Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_two_shots_final, df_breakthrough_boosted_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated (Boosted) Inpatient: ' + str(stats.iqr(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Two Shots vs Boosted Diagnoses Unique Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated (Boosted) Inpatient: ' + str(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated (Boosted) Inpatient: ' + str(stats.iqr(df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Two Shots vs Boosted Diagnoses Total Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_two_shots_final[df_breakthrough_two_shots_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


print('# Number of Inpatients Vaccinated, But Not Boosted Matched: ' + str(len(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Min Number of Unique Inpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.min()))
print('# Median Number of Unique Inpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.median()))
print('# IQR Number of Unique Inpatient Medications Vaccinated, But Not Boosted Matched: ' + str(stats.iqr(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count)))
print('# Max Number of Unique Inpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].unique_inpatient_medication_count.max()))
print('#  \Vaccinated, But Not Boosted Matched vs Boosted Unique Inpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'unique_inpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Outpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_outpatient_medication_count.min()))
print('# Max Number of Unique Outpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_outpatient_medication_count.max()))
print('# Median Number of Unique Outpatient Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_outpatient_medication_count.median()))
print('# IQR Number of Unique Outpatient Medications Vaccinated, But Not Boosted Matched: ' + str(stats.iqr(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_outpatient_medication_count)))
print('# Vaccinated, But Not Boosted Matched vs Boosted Unique Outpatient Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_vaccinated_but_not_boosted_matched_final, df_breakthrough_boosted_final, 'unique_outpatient_medication_count')[1]))
print('\n')

print('# Min Number of Unique Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_medication_count.min()))
print('# Max Number of Unique Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_medication_count.max()))
print('# Median Number of Unique Medications Vaccinated, But Not Boosted Matched: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_medication_count.median()))
print('# IQR Number of Unique Medications Vaccinated, But Not Boosted Matched: ' + str(stats.iqr(df_breakthrough_vaccinated_but_not_boosted_matched_final.unique_medication_count)))
print('# Vaccinated, But Not Boosted Matched vs Boosted Unique Medication Count p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_vaccinated_but_not_boosted_matched_final, df_breakthrough_boosted_final, 'unique_medication_count')[1]))
print('\n')

print('# Min Number of Unique Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_unique_diagnoses.min()))
print('# Max Number of Unique Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_unique_diagnoses.max()))
print('# Median Number of Unique Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_unique_diagnoses.median()))
print('# IQR Number of Unique Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(stats.iqr(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_unique_diagnoses)))
print('# Vaccinated, But Not Boosted Matched vs Boosted Diagnoses Unique Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'n_unique_diagnoses')[1]))
print('\n')

print('# Min Number of Total Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_total_diagnoses.min()))
print('# Max Number of Total Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_total_diagnoses.max()))
print('# Median Number of Total Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_total_diagnoses.median()))
print('# IQR Number of Total Diagnoses Vaccinated, But Not Boosted Matched Inpatient: ' + str(stats.iqr(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1].n_total_diagnoses)))
print('# Vaccinated, But Not Boosted Matched vs Boosted Diagnoses Total Medication Count Inpatient p-value: ' + str(run_mann_whitney_u_test(df_breakthrough_vaccinated_but_not_boosted_matched_final[df_breakthrough_vaccinated_but_not_boosted_matched_final['max_patient_class'] > 1], df_breakthrough_boosted_final[df_breakthrough_boosted_final['max_patient_class'] > 1], 'n_total_diagnoses')[1]))


# check normal distribution of the continuous variables
# maternal age
print('Maternal Age COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.age_at_start_dt, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.age_at_start_dt, line='s')
plt.show()

# pregravid bmi
print('Pregravid BMI COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.pregravid_bmi, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.pregravid_bmi, line='s')
plt.show()

# unique inpatient medication acive ingredient count	
print('Unique Inpatient Medication Count COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.unique_inpatient_medication_count[df_unvaccinated_final['max_patient_class'] > 1], line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.unique_inpatient_medication_count[df_breakthrough_mrna_final['max_patient_class'] > 1], line='s')
plt.show()

# unique outpatient medication acive ingredient count	
print('Unique Outpatient Medication Count COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.unique_outpatient_medication_count, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.unique_outpatient_medication_count, line='s')
plt.show()

# unique medication acive ingredient count	
print('Unique Medication Count COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.unique_medication_count, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.unique_medication_count, line='s')
plt.show()

# total diagnoses count
print('Total Diagnoses Count COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.n_total_diagnoses, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.n_total_diagnoses, line='s')
plt.show()

# unique diagonses count	
print('Unique Diagnoses Count COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.n_unique_diagnoses, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.n_unique_diagnoses, line='s')
plt.show()

# total diagnoses count
print('Gestational Days COVID-positive Quantile Quantile Plot')
qqplot(df_unvaccinated_final.gestational_days, line='s')
plt.show()
qqplot(df_breakthrough_mrna_final.gestational_days, line='s')
plt.show()

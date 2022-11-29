# load environment
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import warnings
import xgboost as xgb

# import packages
from pyspark.sql.functions import unix_timestamp
from scipy import stats
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import shuffle

# set up graph parameters
%matplotlib inline
plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42
np.random.seed(123)
spark.conf.set('spark.sql.execution.arrow.enabled', False)


# define universal variables
DATE_PANDEMIC_START = datetime.date(2020, 3, 5)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.cohort_COVID_pregnancy_functions
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
def get_matching_pairs(df_experimental, df_control, scaler=True):
  if scaler:
      scaler = StandardScaler()
      scaler.fit(df_experimental.append(df_control))
      df_experimental_scaler = scaler.transform(df_experimental)
      df_control_scaler = scaler.transform(df_control)
      nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(df_control_scaler)
      distances, indices = nbrs.kneighbors(df_experimental_scaler)
  
  else:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(df_control)
    distances, indices = nbrs.kneighbors(df_experimental)
  indices = indices.reshape(indices.shape[0])
  matched = df_control.iloc[indices, :]
  return matched
  
  
def drop_non_psm_columns(df):
  df = df \
  .drop('pat_id') \
  .drop('episode_id') \
  .drop('child_episode_id') \
  .drop('sex') \
  .drop('pat_enc_csn_id_collect_list') \
  .drop('ordering_datetime_collect_list') \
  .drop('observation_datetime_collect_list') \
  .drop('result_short_collect_list') \
  .drop('flagged_as_collect_list') \
  .drop('order_name_collect_list') \
  .drop('results_category') \
  .drop('first_ordering_datetime') \
  .drop('instance') \
  .drop('birth_date') \
  .drop('name') \
  .drop('type') \
  .drop('status') \
  .drop('start_date') \
  .drop('end_date') \
  .drop('working_delivery_date') \
  .drop('ob_sticky_note_text') \
  .drop('ob_delivery_episode_type') \
  .drop('ob_delivery_delivery_csn_mom') \
  .drop('ob_delivery_labor_onset_date') \
  .drop('ob_delivery_total_delivery_blood_loss_ml') \
  .drop('number_of_fetuses') \
  .drop('child_type') \
  .drop('child_ob_delivery_episode_type') \
  .drop('child_ob_delivery_delivery_date') \
  .drop('ob_delivery_birth_csn_baby') \
  .drop('ob_delivery_delivery_date') \
  .drop('ob_delivery_record_baby_id') \
  .drop('child_ob_delivery_labor_onset_date') \
  .drop('delivery_birth_comments') \
  .drop('delivery_infant_birth_length_in') \
  .drop('delivery_infant_birth_weight_oz') \
  .drop('ob_history_last_known_living_status') \
  .drop('ob_hx_gestational_age_days') \
  .drop('ob_hx_delivery_site') \
  .drop('ob_hx_delivery_site_comment') \
  .drop('ob_hx_living_status') \
  .drop('ob_hx_outcome') \
  .drop('child_ob_sticky_note_text') \
  .drop('child_ob_delivery_total_delivery_blood_loss_ml') \
  .drop('preterm_category') \
  .drop('ob_delivery_department') \
  .drop('gestational_diabetes_status') \
  .drop('gestational_hypertension_status') \
  .drop('preeclampsia_status') \
  .drop('severe_preeclampsia_status') \
  .drop('iv_drug_user') \
  .drop('delivery_cord_vessels') \
  .drop('GPAL_max') \
  .drop('first_immunization_name') \
  .drop('last_immunization_date') \
  .drop('last_immunization_name') \
  .drop('all_immunization_dates') \
  .drop('all_immunization_names') \
  .drop('diabetes_type_2_status') \
  .drop('SecondaryRUCACode2010')
  return(df)


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
    return 'Unknown'
  if i == 'Hispanic or Latino':
    return 'Hispanic or Latino'
  if i == 'American' or i == 'Samoan' or 'Yupik Eskimo':
    return 'Not Hispanic or Latino'
  if i == 'Filipino' or i == 'Hmong':
    return 'Not Hispanic or Latino'
  if i == 'Sudanese':
    return 'Not Hispanic or Latino'
  if i == 'Patient Refused' or i == 'None':
    return 'Unknown'
  return 'Unknown'


def format_parity(i):
  if i is None:
    return 0
  i = int(i)
  if i == 0 or i == 1:
    return 0
  if i > 1 and i < 5:
    return 1
  if i >= 5:
    return 2
  return 0


def format_gravidity(gravidity):
  if gravidity is None:
    return 0
  gravidity = int(gravidity)
  if gravidity == 0 or gravidity == 1:
    return 0
  elif gravidity > 1 and gravidity < 6:
    return 1
  elif gravidity >= 6:
    return 2
  return 0
    
  
def format_preterm_history(preterm_history, gestational_days):

  if preterm_history is None:
    return 0
  else:
    preterm_history = int(preterm_history)
    if preterm_history == 0 or (preterm_history == 1 and gestational_days < 259):
      return 0
    else:
      return 1
  return 0


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


def encode_bmi(bmi):
  if bmi is None or math.isnan(bmi):
    return -1
  bmi = int(bmi)
  if bmi >= 15 and bmi < 18.5:
    return 0
  if bmi < 25:
    return 1
  if bmi < 30:
    return 2
  if bmi < 35:
    return 3
  if bmi < 40:
    return 4
  return -1


def encode_age(age):
  if age < 25:
    return 0
  if age < 30:
    return 1
  if age < 35:
    return 2
  if age < 40:
    return 3
  if age < 45:
    return 4
  return -1


def encode_ruca(ruca):
  if ruca is None:
    return -1
  if ruca == 'Rural':
    return 0
  if ruca == 'SmallTown':
    return 1
  if ruca == 'Micropolitan':
    return 2
  if ruca == 'Metropolitan':
    return 3
  return -1


def calc_days_from_vaccination(full_vaccination_date, conception_date):
  return (conception_date - full_vaccination_date.date()).days


def encode_vaccination_status_at_conception(first_vaccination_date, full_vaccination_date, conception_date):
  '''
  0 = Unvaccinated at Conception
  1 = Partially Vaccinated at Conception
  2 = Fully Vaccinated at Conception
  '''
  if full_vaccination_date < conception_date:
    return 2
  if first_vaccination_date < conception_date:
    return 1
  return 0


def encode_vaccination_type(vaccination_name):
  '''
  0 = Moderna
  1 = Pfizer
  '''
  if 'MODERNA' in vaccination_name:
    return 0
  else:
    return 1

def calc_days_from_pandemic_start(conception_date):
  return (conception_date - DATE_PANDEMIC_START).days


def handle_missing_bmi(df):
  print('# Percent of patients with pregravid BMI:', str(round(100*(len(df) - df['pregravid_bmi'].isna().sum())/len(df), 1)), '%')
  print('# Imputing median pregravid BMI of', str(round(df['pregravid_bmi'].median(), 2)), '...')
  df['pregravid_bmi'].fillna(df['pregravid_bmi'].median(), inplace = True)
  print('\n')
  return df


def handle_missing_data(df, col):
  if df[col].dtypes != np.number:
    df[col] = df[col].astype(float)
  print('# Percent of patients with', col, ':', str(round(100*(len(df) - df[col].isna().sum())/len(df), 1)), '%')
  print('# Imputing median', col, 'of', str(round(df[col].median(), 2)), '...')
  df[col].fillna(df[col].median(), inplace = True)
  print('\n')
  return df


def impute_missing_data(df):
  #df = handle_missing_data(df, 'RPL_THEMES')
  df = handle_missing_data(df, 'RPL_THEME1')
  df = handle_missing_data(df, 'RPL_THEME2')
  df = handle_missing_data(df, 'RPL_THEME3')
  df = handle_missing_data(df, 'RPL_THEME4')
  df['RPL_THEME1'] = 1-df['RPL_THEME1']  # flip direction so low status is 0 and high status is 1
  df = df.fillna(0)
  return df


def rename_columns(df):
  d = {'age_at_start_dt': 'Age', \
       'chronic_diabetes_with_pregnancy_status': 'Chronic Diabetes', \
       'chronic_hypertension_with_pregnancy_status': 'Chronic Hypertension', \
       'commercial_insurance': 'Commercial Insurance', \
       'days_from_vaccination': 'Days from Vaccination', \
       'ethnic_group': 'Ethnicity, Hispanic', \
       'Gravidity': 'Gravidity', \
       'illegal_drug_user': 'Illicit Drug User', \
       'Parity': 'Parity', \
       'pandemic_timing': 'Pandemic Timing', \
       'pregravid_bmi': 'Pregravid BMI', \
       'Preterm_history': 'Preterm History', \
       'race_asian': 'Race, Asian', \
       'race_black': 'Race, Black', \
       'race_other': 'Race, Other', \
       'race_white': 'Race, White', \
       'RPL_THEME1': 'Socioeconomic Status', \
       'RPL_THEME2': 'Household Composition', \
       'RPL_THEME3': 'Minority Status and Language', \
       'RPL_THEME4': 'Housing Density', \
       'ruca_categorization': 'Rural/Urban Categorization', \
       'smoker': 'Smoker', \
       'vaccination_status_at_conception': 'Vaccination Status at Conception', \
       'vaccination_type': 'Vaccination Type'}
  df.rename(d, axis=1, inplace=True)
  return df


def format_dataframe_for_psm(df):
  dict_white = {'White or Caucasian': 1, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_asian = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 1, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_multiracial = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 1, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_other = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 1, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_black = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 1, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_pacific_islander = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 1, 'American Indian or Alaska Native': 0}
  dict_native_american = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 1}
  dict_ethnic_groups = {'Unknown': -1, 'Hispanic or Latino': 1, 'Not Hispanic or Latino': 0}
  dict_fetal_sex = {None: -1, 'Male': 1, 'Female': 0, 'Other': -1, 'Unknown': -1}
  dict_commercial_insurance = {'Medicaid': 0, 'Medicare': 0, 'Uninsured-Self-Pay': 0, None: 0, 'Other': 0, 'Commercial': 1}
  dict_governmental_insurance = {'Medicaid': 1, 'Medicare': 0, 'Uninsured-Self-Pay': 0, None: 0, 'Other': 0, 'Commercial': 0}
  df = drop_non_psm_columns(df).toPandas()
  for index, row in df.iterrows():
    df.at[index, 'race'] = format_race(row['race'])
    df.at[index, 'ethnic_group'] = format_ethnic_group(row['ethnic_group'])
    df.at[index, 'Preterm_history'] = format_preterm_history(row['Preterm_history'], row['gestational_days'])
  for index, row in df.iterrows():
    df.at[index, 'race_white'] = dict_white[row['race']]
    df.at[index, 'race_asian'] = dict_asian[row['race']]
    df.at[index, 'race_black'] = dict_black[row['race']]
    #df.at[index, 'race_multiracial'] = dict_multiracial[row['race']]
    df.at[index, 'race_other'] = dict_other[row['race']]
    #df.at[index, 'race_pacific_islander'] = dict_pacific_islander[row['race']]
    #df.at[index, 'race_native_american'] = dict_native_american[row['race']]
    df.at[index, 'ethnic_group'] = dict_ethnic_groups[row['ethnic_group']]
    df.at[index, 'ob_hx_infant_sex'] = dict_fetal_sex[row['ob_hx_infant_sex']]
    df.at[index, 'commercial_insurance'] = dict_commercial_insurance[row['insurance']]
    #df.at[index, 'medicaid_insurance'] = dict_governmental_insurance[row['insurance']]
    df.at[index, 'Parity'] = format_parity(row['Parity'])
    df.at[index, 'Gravidity'] = format_gravidity(row['Gravidity'])
    df.at[index, 'Parity'] = format_parity(row['Parity'])
    df.at[index, 'delivery_delivery_method'] = encode_delivery_method(row['delivery_delivery_method'])
    df.at[index, 'pregravid_bmi'] = encode_bmi(row['pregravid_bmi'])
    df.at[index, 'ruca_categorization'] = encode_ruca(row['ruca_categorization'])
    df.at[index, 'days_from_vaccination'] = calc_days_from_vaccination(row['full_vaccination_date'], row['conception_date'])
    df.at[index, 'vaccination_status_at_conception'] = encode_vaccination_status_at_conception(row['first_immunization_date'], row['full_vaccination_date'], row['conception_date'])
    df.at[index, 'vaccination_type'] = encode_vaccination_type(row['full_vaccination_name'])
    df.at[index, 'pandemic_timing'] = calc_days_from_pandemic_start(row['conception_date'])
  df = df.drop(columns=['gestational_days', 'insurance', 'race', 'lmp', 'first_immunization_date', 'full_vaccination_date', 'conception_date', 'full_vaccination_name'])
  df = df.drop(columns=['ob_hx_infant_sex', 'delivery_delivery_method', 'RPL_THEMES'])
  print('Columns used for matching:')
  for col in df.columns:
    print(col)
  print('\n')
  print('\n')
  return(df)


# define cohorts
df_vaccinated_all = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna_expanded_7").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_vaccinated_all.createOrReplaceTempView("vaccinated")

df_experimental = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted AS boosted WHERE boosted.ob_delivery_delivery_date >= date_add(boosted.full_vaccination_date, 168) AND '2021-09-22' < boosted.ob_delivery_delivery_date").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_experimental.createOrReplaceTempView("boosted")
df_experimental = spark.sql(
"""
FROM vaccinated AS v
INNER JOIN boosted AS b
ON v.pat_id = b.pat_id
  AND v.episode_id = b.episode_id
  AND v.child_episode_id = b.child_episode_id
SELECT v.*
""")
print('# Number of women boosted at time of delivery: ' + str(df_experimental.count()))

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 168) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_unvaccinated.createOrReplaceTempView("vaccinated_2shots")
df_control = spark.sql(
"""
FROM vaccinated AS v
INNER JOIN vaccinated_2shots AS v2
ON v.pat_id = v2.pat_id
  AND v.episode_id = v2.episode_id
  AND v.child_episode_id = v2.child_episode_id
SELECT v.*
""")
print('# Number of women vaccinated, but not boosted at time of delivery: ' + str(df_control.count()))


# clean and format data for predictive modeling
# format dataframes for propensity score matching
df_experimental_psm = format_dataframe_for_psm(df_experimental)
print('# Number of people boosted at delivery and delivered after 9-22-21: ' + str(len(df_experimental_psm)))
df_control_psm = format_dataframe_for_psm(df_control)
print('# Number of people vaccinated, but not boosted at delivery who delivered after 9-22-21: ' + str(len(df_control_psm)))


# identify top features predicting who gets vaccinated using machine learning
# format for machine learning
df_experimental_psm['vaccination_status'] = 1
df_control_psm['vaccination_status'] = 0
df_final = shuffle(df_experimental_psm.append(df_control_psm, ignore_index=True), random_state=123)
df_final_imputed = impute_missing_data(df_final.copy())
df_final_imputed.head()


# evaluate distribution of time features between cohorts
print('Overall Stats for Days From Full Vaccination Status To Conception')
print('# Median Days From Full Vaccination Status To Conception: ' + str(statistics.median(df_final.days_from_vaccination)))
print('# IQR Days FromFull Vaccination Status To Conception: ' + str(stats.iqr(df_final.days_from_vaccination)))
print('# Min: ' + (str(df_final.days_from_vaccination.min())))
print('# Max: ' + (str(df_final.days_from_vaccination.max())))
print('\n')

print('Unvaccinated Stats for Days From Full Vaccination Status To Conception')
print('# Median Days From Full Vaccination Status To Conception: ' + str(statistics.median(df_final.loc[df_final['vaccination_status'] == 0].days_from_vaccination)))
print('# IQR Days From Full Vaccination Status To Conception: ' + str(stats.iqr(df_final.loc[df_final['vaccination_status'] == 0].days_from_vaccination)))
print('# Min: ' + (str(df_final.loc[df_final['vaccination_status'] == 0].days_from_vaccination.min())))
print('# Max: ' + (str(df_final.loc[df_final['vaccination_status'] == 0].days_from_vaccination.max())))
print('\n')

print('Vaccinated Stats for Days From Full Vaccination Status To Conception')
print('# Median Days From Full Vaccination Status To Conception: ' + str(statistics.median(df_final.loc[df_final['vaccination_status'] == 1].days_from_vaccination)))
print('# IQR Days From Full Vaccination Status To Conception: ' + str(stats.iqr(df_final.loc[df_final['vaccination_status'] == 1].days_from_vaccination)))
print('# Min: ' + (str(df_final.loc[df_final['vaccination_status'] == 1].days_from_vaccination.min())))
print('# Max: ' + (str(df_final.loc[df_final['vaccination_status'] == 1].days_from_vaccination.max())))
print('\n')

print('# Mann Whitney U Test - Days From Full Vaccination Status:')
print('# p-value: ' + str(stats.mannwhitneyu(df_final.loc[df_final['vaccination_status'] == 1].days_from_vaccination, df_final.loc[df_final['vaccination_status'] == 0].days_from_vaccination)[1]))


print('Overall Stats for Days From Pandemic Start To Conception')
print('# Median Days From Pandemic Start To Conception: ' + str(statistics.median(df_final.pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_final.pandemic_timing)))
print('# Min: ' + (str(df_final.pandemic_timing.min())))
print('# Max: ' + (str(df_final.pandemic_timing.max())))
print('\n')

print('Vaccinated, But Not Boosted Stats for Days From Pandemic Start To Conception')
print('# Median Days From Pandemic Start To Conception: ' + str(statistics.median(df_final.loc[df_final['vaccination_status'] == 0].pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_final.loc[df_final['vaccination_status'] == 0].pandemic_timing)))
print('# Min: ' + (str(df_final.loc[df_final['vaccination_status'] == 0].pandemic_timing.min())))
print('# Max: ' + (str(df_final.loc[df_final['vaccination_status'] == 0].pandemic_timing.max())))
print('\n')

print('Boosted Stats for Days From Pandemic Start To Conception')
print('# Median Days From Pandemic Start To Conception: ' + str(statistics.median(df_final.loc[df_final['vaccination_status'] == 1].pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_final.loc[df_final['vaccination_status'] == 1].pandemic_timing)))
print('# Min: ' + (str(df_final.loc[df_final['vaccination_status'] == 1].pandemic_timing.min())))
print('# Max: ' + (str(df_final.loc[df_final['vaccination_status'] == 1].pandemic_timing.max())))
print('\n')

print('# Mann Whitney U Test - Days From Pandemic Start:')
print('# p-value: ' + str(stats.mannwhitneyu(df_final.loc[df_final['vaccination_status'] == 1].pandemic_timing, df_final.loc[df_final['vaccination_status'] == 0].pandemic_timing)[1]))


# perform pearson correlations
print('Pearson Correlation Parity:')
print(scipy.stats.pearsonr(df_final_imputed.Parity.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Gravidity:')
print(scipy.stats.pearsonr(df_final_imputed.Gravidity.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Preterm History:')
print(scipy.stats.pearsonr(df_final_imputed.Preterm_history.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Commercial Insurance:')
print(scipy.stats.pearsonr(df_final_imputed.commercial_insurance.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Ethnicity:')
print(scipy.stats.pearsonr(df_final_imputed.ethnic_group.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Race, Asian:')
print(scipy.stats.pearsonr(df_final_imputed.race_asian.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Race, Black:')
print(scipy.stats.pearsonr(df_final_imputed.race_black.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Race, Other:')
print(scipy.stats.pearsonr(df_final_imputed.race_other.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Race, White:')
print(scipy.stats.pearsonr(df_final_imputed.race_white.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Maternal Age:')
print(scipy.stats.pearsonr(df_final.age_at_start_dt.to_numpy(), df_final.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Pregravid BMI:')
print(scipy.stats.pearsonr(df_final_imputed.pregravid_bmi.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Smoker:')
print(scipy.stats.pearsonr(df_final_imputed.smoker.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Illicit Drug User:')
print(scipy.stats.pearsonr(df_final_imputed.illegal_drug_user.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Chronic Hypertension Status:')
print(scipy.stats.pearsonr(df_final_imputed.chronic_hypertension_with_pregnancy_status.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Chronic Diabetes Status:')
print(scipy.stats.pearsonr(df_final_imputed.chronic_diabetes_with_pregnancy_status.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation RPL Theme 1 - Socioeconomic Status:')
print(scipy.stats.pearsonr(df_final_imputed.RPL_THEME1.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation RPL Theme 2 - Household Composition:')
print(scipy.stats.pearsonr(df_final_imputed.RPL_THEME2.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation RPL Theme 3 - Minority Status:')
print(scipy.stats.pearsonr(df_final_imputed.RPL_THEME3.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation RPL Theme 4 - Housing and Transportation Status:')
print(scipy.stats.pearsonr(df_final_imputed.RPL_THEME4.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation RUCA Categorization:')
print(scipy.stats.pearsonr(df_final_imputed.ruca_categorization.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Days from Vaccination:')
print(scipy.stats.pearsonr(df_final_imputed.days_from_vaccination.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Vaccination Status at Conception:')
print(scipy.stats.pearsonr(df_final_imputed.vaccination_status_at_conception.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Vaccination Type:')
print(scipy.stats.pearsonr(df_final_imputed.vaccination_type.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')
print('Pearson Correlation Days from Pandemic Start:')
print(scipy.stats.pearsonr(df_final_imputed.pandemic_timing.to_numpy(), df_final_imputed.vaccination_status.to_numpy()))
print('\n')


# declare x and y and scale data
# define
df_model = df_final.copy()
y = df_model.vaccination_status
X_temp = rename_columns(impute_missing_data(df_model.drop(['vaccination_status'], axis=1)))

scaler = StandardScaler()
scaler.fit(X_temp)
# X = pd.DataFrame(scaler.transform(X_temp), columns=X_temp.columns).drop(['Pandemic Timing'], axis=1)
X = pd.DataFrame(scaler.transform(X_temp), columns=X_temp.columns)


# preform variant impact factor analysis
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# declare test/training sets for ml
offset = -1 * int(0.2*len(X))

# Split the data into training/testing sets
X_train = X[:offset]
X_test = X[offset:]

# Split the targets into training/testing sets
y_train = y[:offset]
y_test = y[offset:]


# create and evaluate logistic regression model
# Create 10-fold cross validation object
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Create logistical regression object
regr = LogisticRegression(max_iter=1000)

# Train the model using the training sets
regr.fit(X_train, y_train)

# evaluate model
scores = cross_val_score(regr, X_train, y_train, cv=10)
# report performance
print(scores)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % metrics.mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % metrics.r2_score(y_test, y_pred))

# Evaluate the performance of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Area Under the Curve: ', metrics.roc_auc_score(y_test, y_pred))
print('Coefficient of determination:', metrics.r2_score(y_test, y_pred))

print('\n')
# get importance
predictors = [x for x in X_train.columns]
importance = pd.Series(regr.coef_[0], predictors).sort_values(ascending=False)
print(importance)

# plot feature importance
labels = ['Days from Vaccination', 'Vaccination Status at Conception', 'Age', 'Commercial Insurance', 'Socioeconomic Status', 'Race, Asian', 'Housing Density', 'Vaccination Type', 'Pregravid BMI', 'Illicit Drug User', 'Race, White', 'Rural/Urban Categorization', 'Chronic Diabetes', 'Chronic Hypertension', 'Race, Other', 'Preterm History', 'Parity', 'Minority Status and Language', 'Race, Black', 'Smoker', 'Ethnicity, Hispanic', 'Gravidity', 'Household Composition', 'Pandemic Timing']
plt.bar([x for x in range(len(importance))], importance, color='mediumorchid')
plt.xticks(np.arange(0, 24, 1), labels=labels, rotation = 90)
plt.ylabel('Contribution')
plt.title('Logistic Regression')
display(plt.show())


# create and evaluate perfomance of XGBoost model
# Initiate, train and evaluate Gradient Boosting Regressor Model
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train,y_train)
xgb_classifier.predict(X_test[1:2])
xgb_classifier.score(X_test, y_test)

# evaluate model
scores = cross_val_score(xgb_classifier, X_train, y_train, cv=10)
# report performance
print(scores)

# Make predictions using the testing set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the performance of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Area Under the Curve: ', metrics.roc_auc_score(y_test, y_pred))
print('Coefficient of determination:', metrics.r2_score(y_test, y_pred))

print('\n')
# get importance
predictors = [x for x in X_train.columns]
importance = pd.Series(xgb_classifier.feature_importances_, predictors).sort_values(ascending=False)
print(importance)

# plot feature importance
labels = ['Commercial Insurance', 'Days from Vaccination', 'Pandemic Timing', 'Race, Asian', 'Ethnicity, Hispanic', 'Illicit Drug User', 'Chronic Hypertension', 'Chronic Diabetes', 'Age', 'Smoker', 'Socioeconomic Status', 'Household Composition', 'Preterm History', 'Minority Status and Language', 'Gravidity', 'Rural/Urban Categorization', 'Race, Black', 'Housing Density', 'Vaccination Type', 'Race, Other', 'Race, White', 'Pregravid BMI', 'Parity', 'Vaccination Status at Conception']
plt.bar([x for x in range(len(importance))], importance, color='chocolate')
plt.xticks(np.arange(0, 24, 1), labels=labels, rotation = 90)
plt.ylabel('Contribution')
plt.title('XGBoost')
display(plt.show())


# create and evaluate performance of random forest model
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)

# evaluate model
scores = cross_val_score(rf, X_train, y_train, cv=10)
# report performance
print(scores)

# Use the forest's predict method on the test data
y_pred = rf.predict(X_test)

# Evaluate the performance of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Area Under the Curve: ', metrics.roc_auc_score(y_test, y_pred))
print('Coefficient of determination:', metrics.r2_score(y_test, y_pred))

print('\n')
# get importance
predictors = [x for x in X_train.columns]
importance = pd.Series(rf.feature_importances_, predictors).sort_values(ascending=False)
print(importance)

# plot feature importance
labels = ['Days from Vaccination', 'Pandemic Timing', 'Age', 'Household Composition', 'Minority Status and Language', 'Housing Density', 'Socioeconomic Status', 'Commercial Insurance', 'Pregravid BMI', 'Gravidity', 'Vaccination Type', 'Parity', 'Race, White', 'Ethnicity, Hispanic', 'Race, Asian', 'Chronic Diabetes', 'Rural/Urban Categorization', 'Race, Other', 'Illicit Drug User', 'Smoker', 'Preterm History', 'Race, Black', 'Chronic Hypertension', 'Vaccination Status at Conception']
plt.bar([x for x in range(len(importance))], importance, color='seagreen')
plt.xticks(np.arange(0, 24, 1), labels=labels, rotation = 90)
plt.ylabel('Contribution')
plt.title('Random Forest')
display(plt.show())


# create and evaluate performance of Gradient Boosting Regression model
# Initiate, train and evaluate Gradient Boosting Regressor Model
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
GradientBoostingRegressor(random_state=0)
reg.predict(X_test[1:2])
reg.score(X_test, y_test)

# evaluate model
scores = cross_val_score(reg, X_train, y_train, cv=10)
# report performance
print(scores)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# Evaluate the performance of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Area Under the Curve: ', metrics.roc_auc_score(y_test, y_pred))
print('Coefficient of determination:', metrics.r2_score(y_test, y_pred))

print('\n')
# get importance
predictors = [x for x in X_train.columns]
importance = pd.Series(reg.feature_importances_, predictors).sort_values(ascending=False)
print(importance)

# calculate the fpr and tpr for all thresholds of the classification
fpr_original, tpr_original, threshold_original = metrics.roc_curve(y_test, y_pred)
roc_auc_original = metrics.auc(fpr_original, tpr_original)

# plot feature importance
labels = ['Days from Vaccination', 'Pandemic Timing', 'Commercial Insurance', 'Age', 'Household Composition', 'Socioeconomic Status', 'Minority Status and Language', 'Ethnicity, Hispanic', 'Housing Density', 'Race, Asian', 'Gravidity', 'Vaccination Type', 'Pregravid BMI', 'Illicit Drug User', 'Chronic Hypertension', 'Race, Black', 'Smoker', 'Chronic Diabetes', 'Parity', 'Preterm History', 'Rural/Urban Categorization', 'Race, White', 'Race, Other', 'Vaccination Status at Conecption']
plt.bar([x for x in range(len(importance))], importance, color='hotpink')
plt.xticks(np.arange(0, 24, 1), labels=labels, rotation = 90)
plt.ylabel('Contribution')
plt.title('Gradient Boosting Regression')
display(plt.show())

with warnings.catch_warnings():
  warnings.filterwarnings("ignore")
  reg_explainer = shap.KernelExplainer(reg.predict, shap.sample(X_test, 1000))
  reg_shap_values = reg_explainer.shap_values(shap.sample(X_test, 1000))
  shap.summary_plot(reg_shap_values, shap.sample(X_test, 1000))


# preform variant impact factor analysis for the limited model
X = X[['Age', 'Pandemic Timing',  'Days from Vaccination', 'Commercial Insurance', 'Household Composition']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# create and evaluate performance of Gradient Boosting Regression Limited Model
# Subset test and training set to only include a limited number of variables
X_train = X_train[['Age', 'Pandemic Timing',  'Days from Vaccination', 'Commercial Insurance', 'Household Composition']]
X_test = X_test[['Age', 'Pandemic Timing',  'Days from Vaccination', 'Commercial Insurance', 'Household Composition']]

# Initiate, train and evaluate Gradient Boosting Regressor Model
reg_limited = GradientBoostingRegressor(random_state=0)
reg_limited.fit(X_train, y_train)
GradientBoostingRegressor(random_state=0)
reg_limited.predict(X_test[1:2])
reg_limited.score(X_test, y_test)

# evaluate model
scores = cross_val_score(reg_limited, X_train, y_train, cv=10)
# report performance
print(scores)

# Make predictions using the testing set
y_pred = reg_limited.predict(X_test)

# evaluate model performace
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Area Under the Curve: ', metrics.roc_auc_score(y_test, y_pred))
print('Coefficient of determination:', metrics.r2_score(y_test, y_pred))

print('\n')
# get importance
predictors = [x for x in X_train.columns]
importance = pd.Series(reg_limited.feature_importances_, predictors).sort_values(ascending=False)
print(importance)

# plot feature importance
labels = ['Days from\nVaccination', 'Pandemic\nTiming', 'Commercial\nInsurance', 'Age', 'Household\nComposition']
plt.figure(figsize=(5, 7))
plt.bar([x for x in range(len(importance))], importance, color='c')
plt.xticks(np.arange(0, 5, 1), labels=labels, rotation = 90)
plt.ylabel('Contribution')
display(plt.show())

with warnings.catch_warnings():
  warnings.filterwarnings("ignore")
  reg_explainer = shap.KernelExplainer(reg_limited.predict, shap.sample(X_test, 1000))
  reg_shap_values = reg_explainer.shap_values(shap.sample(X_test, 1000))
  shap.summary_plot(reg_shap_values, shap.sample(X_test, 1000))
  

# plot AUC of Complete and Limited Gradient Boosting Regression Models
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.figure(figsize=(5, 7))
plt.title('Vaccination Status')
plt.plot(fpr_original, tpr_original, 'b', label = 'Original Model AUC = %0.2f' % roc_auc_original, color='hotpink')
plt.plot(fpr, tpr, 'b', label = 'Limited Model AUC = %0.2f' % roc_auc, color='c')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--', color='k')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

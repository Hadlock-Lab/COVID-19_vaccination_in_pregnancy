# install nad import packages
dbutils.library.installPyPI("rpy2")
import pandas as pd
import numpy as np
import rpy2
import scipy
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import date, datetime
from pyspark.sql.functions import rand, unix_timestamp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.utils import shuffle
import math
!pip install psmpy
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42
np.random.seed(123)
spark.conf.set('spark.sql.execution.arrow.enabled', False)


# set universal parameters
DATE_PANDEMIC_START = date(2020, 3, 5)

# import functions from other notebooks
%run
"Cohort/Cohort_COVID_Pregnancy_Functions"
%run
"Utilities/sars_cov_2_cohort_v2"


# define functions
def get_matching_pairs(df_experimental, df_control, n, scaler=True):
  if scaler:
      scaler = StandardScaler()
      scaler.fit(df_experimental.append(df_control))
      df_experimental_scaler = scaler.transform(df_experimental)
      df_control_scaler = scaler.transform(df_control)
      nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric='euclidean').fit(df_control_scaler)
      distances, indices = nbrs.kneighbors(df_experimental_scaler)
  
  else:
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric='euclidean').fit(df_control)
    distances, indices = nbrs.kneighbors(df_experimental)
  indices = indices.reshape(indices.shape[0])
  matched = df_control.iloc[indices, :]
  return matched
  
  
def drop_non_psm_columns(df):
  df = df \
  .drop('sex') \
  .drop('pat_enc_csn_id_collect_list') \
  .drop('ordering_datetime_collect_list') \
  .drop('observation_datetime_collect_list') \
  .drop('result_short_collect_list') \
  .drop('flagged_as_collect_list') \
  .drop('order_name_collect_list') \
  .drop('results_category') \
  .drop('first_ordering_datetime') \
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
  print('#b Imputing median', col, 'of', str(round(df[col].median(), 2)), '...')
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
    #df.at[index, 'age_at_start_dt'] = encode_age(row['age_at_start_dt'])
  df = df.drop(columns=['gestational_days', 'insurance', 'race', 'lmp', 'first_immunization_date', 'full_vaccination_date', 'conception_date', 'full_vaccination_name', 'conception_date', 'first_immunization_date', 'full_vaccination_date', 'full_vaccination_name'])
  df = df.drop(columns=['ob_hx_infant_sex', 'delivery_delivery_method', 'RPL_THEMES'])
  df = impute_missing_data(df)
  print('Columns used for matching:')
  for col in df.columns:
    print(col)
  print('\n')
  print('\n')
  return(df)


def retrieve_matched_id_info(df_final, df_psm):
  df_matched = pd.DataFrame(columns=list(df_final.columns))
  for item in list(df_psm['matched_ID']):
    row = df_final.loc[df_final['id'] == item]
    df_matched = df_matched.append(row, ignore_index=True)
  print('# Number of Matched Unvaccinated Patients: ' + str(len(df_matched)))
  return df_matched


def calc_fishers_exact_test(dict_obs, dict_exp):
  import numpy as np
  import rpy2.robjects.numpy2ri
  from rpy2.robjects.packages import importr
  rpy2.robjects.numpy2ri.activate()

  stats = importr('stats')
  list_obs, list_exp = [], []
  total_obs = sum(dict_obs.values())
  total_exp = sum(dict_exp.values())
  for key in dict_obs.keys():
    list_obs.append(dict_obs[key])
    list_exp.append(dict_exp[key])
  list_contingency = np.array([list_obs, list_exp])
  print(list_contingency)
  res = stats.fisher_test(x=list_contingency, simulate_p_value=True)
  print(res)
  
  
def add_item_to_dict(d, k):
  if k in d:
    d[k]+=1
  else:
    d[k] = 1
  return d


def print_dict(d, count):
  for k, v in d.items():
    print(str(k) + ': ' + str(v) + ' (' + str(round(v/count, 3)*100) + '%)')


def make_dict(l):
  d = {}
  l = list(l)
  for i in l:
    d = add_item_to_dict(d, i)
  print_dict(d, len(l))
  return d


def retrieve_matched_id_info(df_final, df_psm):
  df_matched = pd.DataFrame(columns=list(df_final.columns))
  for item in list(df_psm['matched_ID']):
    row = df_final.loc[df_final['id'] == item]
    df_matched = df_matched.append(row, ignore_index=True)
  print('# Number of Matched Unvaccinated Patients: ' + str(len(df_matched)))
  return df_matched


# define cohorts
# set table_name for saving final matched control table
table_name = 'snp3_cohort_maternity_vaccinated_but_not_boosted_matched_control'

# load saved dataframes
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
df_experimental = df_experimental.orderBy(rand()).limit(4414)
print('# Number of randomly selected women boosted at time of delivery for propensity score matching: ' + str(df_experimental.count()))

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 168) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
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


# format dataframes for propensity score matching
df_experimental_psm = format_dataframe_for_psm(df_experimental)
print('# Number of women vaccinated at time of delivery: ' + str(len(df_experimental_psm)))
df_control_psm = format_dataframe_for_psm(df_control)
print('# Number of unvaccinated pregnant women who have delivered after 1-25-21: ' + str(len(df_control_psm)))


# identify top features predicting who gets vaccinated using machine learning
# format for machine learning
df_experimental_psm['vaccination_status'] = 1
df_control_psm['vaccination_status'] = 0
df_final = shuffle(df_experimental_psm.append(df_control_psm, ignore_index=True), random_state=123)
cols = df_final.columns
df_final[cols] = df_final[cols].apply(pd.to_numeric, errors='coerce')
df_final = df_final.dropna()
df_final['id'] = df_final['pat_id'].astype(str) + df_final['episode_id'].astype('int').astype('str') + df_final['child_episode_id'].astype('int').astype('str')
df_final[cols] = df_final[cols].apply(pd.to_numeric, errors='coerce')


# propensity score match with replacement unvaccinated with vaccinated patients
psm = PsmPy(df_final, treatment='vaccination_status', indx='id', exclude = ['pat_id', 'instance', 'episode_id', 'child_episode_id'])
# psm = PsmPy(df_final, treatment='vaccination_status', indx='id', exclude = ['pat_id', 'instance', 'episode_id', 'child_episode_id', 'ethnic_group', 'chronic_diabetes_with_pregnancy_status', 'chronic_hypertension_with_pregnancy_status', 'smoker', 'illegal_drug_user', 'Parity', 'Gravidity', 'Preterm_history', 'race_white', 'race_black', 'race_other', 'RPL_THEME1', 'RPL_THEME3', 'RPL_THEME4'])
psm.logistic_ps(balance=True)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.6)


# evaluate propensity score performance
psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity logit', names = ['Vaccinated', 'Unvaccinated'], save=True)
psm.effect_size_plot(save=False)
psm.effect_size_plot(save=False)


# retrieve original dataframe info for matched people
df_matched = retrieve_matched_id_info(df_final, psm.matched_ids)
df_temp = spark.createDataFrame(df_matched[['pat_id', 'instance', 'episode_id', 'child_episode_id']])
df_control = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna_expanded_7 AS mrna WHERE mrna.ob_delivery_delivery_date >= date_add(mrna.full_vaccination_date, 168) AND '2021-09-22' < mrna.ob_delivery_delivery_date").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_control.createOrReplaceTempView("control")
df_temp.createOrReplaceTempView("temp")
df_matched_final = spark.sql(
"""
SELECT c.*
FROM control AS c
INNER JOIN temp AS t 
ON c.pat_id = t.pat_id
  AND c.instance = t.instance
  AND c.episode_id = t.episode_id
  AND c.child_episode_id = t.child_episode_id
  """)
print('# Number of pregnancies in matched control vaccinated, but not boosted cohort: ' + str(df_matched_final.count()))
print('# Number of unique pregnancies in matched control vaccinated but not boosted cohort: ' + str(df_matched_final.dropDuplicates().count()))


# save matched dataframe
write_data_frame_to_sandbox(df_matched_final, table_name, sandbox_db='rdp_phi_sandbox', replace=True)


# Identify unvaccinated match controls that had a covid-19 infection
df_covid_mom = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_test_data")
df_covid_mom.createOrReplaceTempView("covid_mom")
df_matched_final.createOrReplaceTempView("matched")
df_matched_covid = spark.sql(
"""
FROM matched AS m
INNER JOIN covid_mom AS cm
ON m.pat_id = cm.pat_id
SELECT m.*, cm.covid_induced_immunity, cm. covid_test_result, cm.covid_test_date, cm.covid_test_number_of_days_from_conception, cm.trimester_of_covid_test
""")
write_data_frame_to_sandbox(df_matched_covid, table_name + '_covid', sandbox_db='rdp_phi_sandbox', replace=True)

print('# Number of pregnancies in matched control vaccinated but not boosted cohort during which a covid infection occured: ' + str(df_matched_covid.count()))
print('# Number of unique pregnancies in matched control vaccinated but not boosted cohort during which a covid infection occured: ' + str(df_matched_covid.dropDuplicates().count()))



# load and format dataframes
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
df_experimental_psm = format_dataframe_for_psm(df_experimental)

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 168) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
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
df_control_psm = format_dataframe_for_psm(df_control)

df_matched_final = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_but_not_boosted_matched_control")
df_matched_final_psm = format_dataframe_for_psm(df_matched_final)


# evaluate vaccine related features in the propensity score matched cohort
print('# Unvaccinated Stats for Days From Pandemic Start To Conception')
print('# Median Days From Pandemic Start To Conception: ' + str(statistics.median(df_control_psm.pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_control_psm.pandemic_timing)))
print('# Min: ' + (str(df_control_psm.pandemic_timing.min())))
print('# Max: ' + (str(df_control_psm.pandemic_timing.max())))
print('\n')

print('# Vaccinated Stats for Days Pandemic Start To Conception')
print('# Median Days From Pandemic Start To Conception: ' + str(statistics.median(df_experimental_psm.pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_experimental_psm.pandemic_timing)))
print('# Min: ' + (str(df_experimental_psm.pandemic_timing.min())))
print('# Max: ' + (str(df_experimental_psm.pandemic_timing.max())))
print('\n')

print('# Mann Whitney U Test - Days From Pandemic Start to Conception:')
print('# p-value: ' + str(stats.mannwhitneyu(df_experimental_psm.pandemic_timing, df_control_psm.pandemic_timing)[1]))


print('# Unvaccinated Matched Stats for Days From Pandemic Start To Conception')
print('# Median Days FromPandemic Start To Conception: ' + str(statistics.median(df_matched_final_psm.pandemic_timing)))
print('# IQR Days From Pandemic Start To Conception: ' + str(stats.iqr(df_matched_final_psm.pandemic_timing)))
print('# Min: ' + (str(df_matched_final_psm.pandemic_timing.min())))
print('# Max: ' + (str(df_matched_final_psm.pandemic_timing.max())))
print('\n')

print('# Mann Whitney U Test - Days From Pandemic Start to Conception:')
print('# p-value: ' + str(stats.mannwhitneyu(df_experimental_psm.pandemic_timing, df_matched_final_psm.pandemic_timing)[1]))

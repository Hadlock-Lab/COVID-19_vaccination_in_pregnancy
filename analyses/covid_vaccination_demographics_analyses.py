Author: Samantha Piekos
Date: 11/14/22


# load environment
import datetime
import numpy as np
import pandas as pd
import statistics

from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed
from pyspark.sql.functions import col, lit, SparkSession, unix_timestamp
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

!pip install rpy2
import rpy2

seed(31415)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
def print_dict(d, count):
  for k, v in d.items():
    print(str(k) + ': ' + str(v) + ' (' + str(round(v/count, 3)*100) + '%)')
  print('\n')


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
  print_dict(dict_obs, sum(dict_obs.values()))
  print_dict(dict_exp, sum(dict_exp.values()))
  res = stats.fisher_test(x=list_contingency, simulate_p_value=True)
  print(res)
  
  
def calc_fishers_exact_test_not_simulated(dict_obs, dict_exp):
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
  
def calc_chi_square(dict_obs, dict_exp):
  from scipy.stats import chisquare
  list_obs, list_exp = [], []
  total_obs = sum(dict_obs.values())
  total_exp = sum(dict_exp.values())
  for key in dict_obs.keys():
    list_obs.append(dict_obs[key])
    list_exp.append(dict_exp[key])
  print(list_obs, list_exp)
  chi_test = chisquare(f_obs=list_obs, f_exp=list_exp)
  print(chi_test)

def calc_chi_square_from_dataframe(df_1, df_2, k):
  dict_1, dict_2 = {}, {}
  df_1_sum = sum(df_1[k])
  dict_1['yes'] = (sum(df_1[k]))
  dict_1['no'] = (len(df_1[k])-df_1_sum)
  df_2_sum = sum(df_2[k])
  dict_2['yes'] = (sum(df_2[k]))
  dict_2['no'] = (len(df_2[k])-df_2_sum)
  calc_chi_square(dict_1, dict_2)
  

def get_races(df):
  import pandas as pd
  dict_races = {'Multiracial': 0}
  for index, item in df.iterrows():
    for i in item:
      for entry in i:
        print(entry)
  print_races
  return(dict_races)


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


def determine_race_distribution(df):
  import pandas as pd
  dict_races = {'Multiracial': 0, 'White or Caucasian': 0, 'Other': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'Unknown': 0, 'Asian': 0, 'Black or African American': 0,\
                'American Indian or Alaska Native': 0}
  c = 0
  for index, i in df.iteritems():
    c += 1
    if i is None:
      dict_races['Unknown'] += 1
      continue
    elif isinstance(i, int):
      if i == 0:
        dict_races['Unknown'] += 1
        continue
    elif len(i) > 1:
      dict_races[handle_multiracial_exceptions(i)] += 1
      continue
    else:
      if i[0] == 'White':
        i[0] = 'White or Caucasian'
      elif i[0] == 'Unable to Determine' or i[0] == 'Patient Refused' or i[0] == 'Declined':
        i[0] = 'Unknown'
      dict_races[i[0]] += 1
      continue
  print('Total Number of Patients: ' + str(c))
  print('\n')
  for key, value in dict_races.items():
    print(key + ': ' + str(value))
  return(dict_races)


def determine_ethnic_distribution_2(df):
  import pandas as pd
  dict_ethnic_groups = {'Unknown': 0, 'Hispanic or Latino': 0, 'Not Hispanic or Latino': 0}
  c = 0
  for index, i in df.iteritems():
    c += 1
    if i is None:
      dict_ethnic_groups['Unknown'] += 1
    elif isinstance(i, int):
      if i == 0:
        dict_ethnic_groups['Unknown'] += 1
    elif i == 'Hispanic or Latino':
      dict_ethnic_groups['Hispanic or Latino'] += 1
    elif i == 'American' or i == 'Samoan' or 'Yupik Eskimo':
      dict_ethnic_groups['Not Hispanic or Latino'] += 1
    elif i == 'Filipino' or i == 'Hmong':
      dict_ethnic_groups['Not Hispanic or Latino'] += 1
    elif i == 'Sudanese':
      dict_ethnic_groups['Not Hispanic or Latino'] += 1
    elif i == 'None' or i == 'Patient Refused':
      dict_ethnic_groups['Unknown'] += 1
    else:
      dict_ethnic_groups['Unknown'] += 1
  print('Total Number of Patients: ' + str(c))
  print('\n')
  for key, value in dict_ethnic_groups.items():
    print(key + ': ' + str(value))
  return(dict_ethnic_groups )


def map_to_preterm_labels(gestational_age):
    if(gestational_age is None):
      return None
    elif(gestational_age >= 259):
      return 'term'
    elif(gestational_age >= 238):
      return 'late_preterm'
    elif(gestational_age >= 224):
      return 'moderate_preterm'
    elif(gestational_age >= 196):
      return 'very_preterm'
    else:
      return 'extremely_preterm'
    
    
def determine_preterm_status(df):
  dict_preterm_status = {'term': 0, 'late_preterm': 0, 'moderate_preterm': 0, 'very_preterm': 0, 'extremely_preterm': 0}
  for index, row in df.iterrows():
    gestational_days = row['gestational_days']
    preterm_status = map_to_preterm_labels(gestational_days)
    dict_preterm_status[preterm_status] += 1
  print_dict(dict_preterm_status, len(df))
  return dict_preterm_status


def determine_term_or_preterm_status(df):
  d = {'term': 0, 'preterm': 0}
  for index, row in df.iterrows():
    gestational_days = row['gestational_days']
    if gestational_days >= 259:
      d['term'] += 1
    else:
      d['preterm'] += 1
  print_dict(d, len(df))
  return d


def add_to_dict(d, k):
  if k not in d.keys():
    d[k] = 1
  else:
    d[k] += 1
  return d


def determine_count(df, k):
  d = {}
  for index, row in df.iterrows():
    if row[k] is not None:
      d = add_to_dict(d, row[k])
    else:
      d = add_to_dict(d, 'Missing')
  for key, value in d.items():
    print(str(key) + ': ' + str(value))
  return(d)


def make_histogram(df, k, bin_size, minimum, maximum, color='cornflowerblue'):
  data = []
  for index, row in df.iterrows():  # make list of value of interest
    if row[k] is not None and row[k] != "-999.0":
      data.append(float(row[k]))
  # plot histogram
  bins = np.arange(-1000, 1000, bin_size) # fixed bin size

  plt.xlim([minimum, maximum])

  plt.hist(data, bins=bins, alpha=0.5, color=color)
  plt.xlabel(k + ' (Bin Size = ' + str(bin_size) + ')')
  plt.ylabel('Count')
  display(plt.show())
  
  
def make_histogram_2(data, k, bin_size, minimum, maximum, color='cornflowerblue'):
  # plot histogram from list
  bins = np.arange(-1000, 1000, bin_size) # fixed bin size

  plt.xlim([minimum, maximum])

  plt.hist(data, bins=bins, alpha=0.5, color=color)
  plt.xlabel(k + ' (Bin Size = ' + str(bin_size) + ')')
  plt.ylabel('Count')
  display(plt.show())
  
  
def run_t_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None and row[k] != "-999.0":
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None and row[k] != "-999.0":
      data_2.append(float(row[k]))
  return(stats.ttest_ind(data_1, data_2))


def run_mann_whitney_u_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None and row[k] != "-999.0":
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None and row[k] != "-999.0":
      data_2.append(float(row[k]))
  return(stats.mannwhitneyu(data_1, data_2))


def calc_chi_square_for_delivery_method(d_1, d_2):
  d_clean_1, d_clean_2 = {'Vaginal': 0, 'C-Section': 0}, {'Vaginal': 0, 'C-Section': 0}
  for key in d_1.keys():
    if 'Vaginal' in key:
      d_clean_1['Vaginal'] += d_1[key]
    elif 'C-Section' in key:
      d_clean_1['C-Section'] += d_1[key]
  for key in d_2.keys():
    if 'Vaginal' in key:
      d_clean_2['Vaginal'] += d_2[key]
    elif 'C-Section' in key:
      d_clean_2['C-Section'] += d_2[key]
  print_dict(d_clean_1, sum(d_clean_1.values()))
  print_dict(d_clean_2, sum(d_clean_2.values()))
  return(calc_chi_square(d_clean_1, d_clean_2))


def calc_fishers_exact_test_for_delivery_method(d_1, d_2):
  d_clean_1, d_clean_2 = {'Vaginal': 0, 'C-Section': 0}, {'Vaginal': 0, 'C-Section': 0}
  for key in d_1.keys():
    if 'Vaginal' in key:
      d_clean_1['Vaginal'] += d_1[key]
    elif 'C-Section' in key:
      d_clean_1['C-Section'] += d_1[key]
  for key in d_2.keys():
    if 'Vaginal' in key:
      d_clean_2['Vaginal'] += d_2[key]
    elif 'C-Section' in key:
      d_clean_2['C-Section'] += d_2[key]
  print_dict(d_clean_1, sum(d_clean_1.values()))
  print_dict(d_clean_2, sum(d_clean_2.values()))
  return(calc_fishers_exact_test_not_simulated(d_clean_1, d_clean_2))


def determine_insurance_distribution(df):
  import pandas as pd
  dict_insurance = {'Medicaid': 0, 'Commercial': 0, 'Uninsured-Self-Pay': 0, 'Medicare': 0}
  c = 0
  for index, item in df.iterrows():
    insurance = item['insurance']
    if insurance is None:
      dict_insurance['Uninsured-Self-Pay'] += 1
    elif isinstance(insurance, int):
      if insurance == 0:
        dict_insurance['Uninsured-Self-Pay'] += 1
    elif insurance == 'Other':
      dict_insurance['Uninsured-Self-Pay'] += 1
    else:
      dict_insurance[insurance] += 1
  print('Total Number of Patients with Medicaid: ' + str(dict_insurance['Medicaid']))
  print('Total Number of Patients with commercial insurance: ' + str(dict_insurance['Commercial']))
  print('Total Number of Patients with uninsured/self-pay insurance: ' + str(dict_insurance['Uninsured-Self-Pay']))
  print('Total Number of Patients with Medicare insurance: ' + str(dict_insurance['Medicare']))
  return(dict_insurance)


def determine_preterm_history(df):
  dict_preterm_history = {'No': 0, 'Yes': 0}
  for index, item in df.iterrows():
    preterm_history = str(item['Preterm_history'])
    if preterm_history.isnumeric():
      preterm_history = int(preterm_history)
      if preterm_history == 0 or (preterm_history == 1 and item['gestational_days'] < 259):
        dict_preterm_history['No'] += 1
      else:
        dict_preterm_history['Yes'] += 1
    else:
      dict_preterm_history['No'] += 1
  print('Total Number of Patients With History of Pretem Birth: ' + str(dict_preterm_history['Yes']))
  print('Total Number of Patients Without History of Pretem Birth: ' + str(dict_preterm_history['No']))
  return(dict_preterm_history)


def determine_parity(df):
  dict_parity = {'Nulliparity': 0, 'Low Multiparity': 0, 'Grand Multipara': 0}
  for index, item in df.iterrows():
    parity = str(item['Parity'])
    if parity.isnumeric():
      parity = int(parity)
      if parity == 0 or parity == 1:
        dict_parity['Nulliparity'] += 1
      elif parity > 1 and parity < 5:
        dict_parity['Low Multiparity'] += 1
      else:
        dict_parity['Grand Multipara'] += 1
    else:
      dict_parity['Nulliparity'] += 1
  print('Total Number of Patients With Nulliparity: ' + str(dict_parity['Nulliparity']))
  print('Total Number of Patients With Low Multiparity: ' + str(dict_parity['Low Multiparity']))
  print('Total Number of Patients With Grand Multipara: ' + str(dict_parity['Grand Multipara']))
  return(dict_parity)


def determine_gravidity(df):
  dict_gravidity = {'Primagravida': 0, 'Low Multigravidia': 0, 'Grand Multigravidia': 0}
  for index, item in df.iterrows():
    gravidity = str(item['Gravidity'])
    if gravidity.isnumeric():
      gravidity = int(gravidity)
      if gravidity == 0 or gravidity == 1:
        dict_gravidity['Primagravida'] += 1
      elif gravidity > 1 and gravidity < 6:
        dict_gravidity['Low Multigravidia'] += 1
      else:
        dict_gravidity['Grand Multigravidia'] += 1
    else:
      dict_gravidity['Primagravida'] += 1
  print('Total Number of Patients With Primagravida: ' + str(dict_gravidity['Primagravida']))
  print('Total Number of Patients With Low Multigravidia: ' + str(dict_gravidity['Low Multigravidia']))
  print('Total Number of Patients With Grand Multigravidia: ' + str(dict_gravidity['Grand Multigravidia']))
  return(dict_gravidity)


def get_fetal_growth_percentiles(df):
  data = []
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    gestational_age = row['gestational_days']
    gender = row['ob_hx_infant_sex']
    if gender is None:
      gender = 'unknown'
    if weight is None or math.isnan(weight):
      continue
    if gestational_age is None or math.isnan(gestational_age):
      continue
    data.append(calc_birth_weight_percentile(weight, gestational_age, gender))
  return(data)


def count_small_for_gestational_age(l):
  d = {'SGA': 0, 'normal': 0}
  for item in l:
    if float(item) <= 10:
      d['SGA'] += 1
    else:
      d['normal'] += 1
  print_dict(d, len(l))
  return d


def count_trimester_of_infection(l):
  l = list(l)
  d = {'1st trimester': 0, '2nd trimester': 0, '3rd trimester': 0}
  for i in l:
    d[i] += 1
  print_dict(d, len(l))
  return d


def make_low_birth_weight_dict(df):
  d = {'low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 88.1849:
      d['low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print_dict(d, len(df))
  return d


def make_extremely_low_birth_weight_dict(df):
  d = {'extremely_low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 52.91094:
      d['extremely_low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print_dict(d, len(df))
  return d


def encode_bmi(bmi):
  if bmi is None or math.isnan(bmi):
    return 'Missing'
  bmi = int(bmi)
  if bmi >= 15 and bmi < 18.5:
    return 'Underweight'
  if bmi < 25:
    return 'Normal'
  if bmi < 30:
    return 'Overweight'
  if bmi < 35:
    return 'Obese'
  if bmi < 40:
    return 'Severely Obese'
  return 'Missing'


def make_bmi_dict(df):
  d = {'Missing': 0, 'Underweight': 0, 'Normal': 0, 'Overweight': 0, 'Obese': 0, 'Severely Obese': 0}
  for bmi in df['pregravid_bmi']:
    bmi_category = encode_bmi(bmi)
    d[bmi_category] += 1
  for key, value in d.items():
    print(key + ': ' + str(value))
  print('\n')
  return d


def encode_age(age):
  if age < 25:
    return '18-25'
  if age < 30:
    return '25-30'
  if age < 35:
    return '30-35'
  if age < 40:
    return '35-40'
  if age < 45:
    return '40-45'
  return 'Missing'


def make_age_dict(df):
  d = {'Missing': 0, '18-25': 0, '25-30': 0, '30-35': 0, '35-40': 0, '40-45': 0}
  for age in df['age_at_start_dt']:
    age_category = encode_age(age)
    d[age_category] += 1
  for key, value in d.items():
    print(key + ': ' + str(value))
  print('\n')
  return d


def determine_fetal_sex(df):
  import pandas as pd
  dict_fetal_sex = {'Male': 0, 'Female': 0, 'Unknown': 0}
  c = 0
  for index, item in df.iterrows():
    fetal_sex = item['ob_hx_infant_sex']
    if fetal_sex is None or fetal_sex == 'Other':
      dict_fetal_sex['Unknown'] += 1
      continue
    elif isinstance(fetal_sex, int):
      if fetal_sex == 0:
        dict_fetal_sex['Unknown'] += 1
        continue
    dict_fetal_sex[fetal_sex] += 1
  return(dict_fetal_sex)


def evaluate_quintile(d, i):
  i = float(i)
  if i < 0:
    d['Missing'] += 1
  if i < 0.2:
    d['1st Quintile'] += 1
  elif i < 0.4:
    d['2nd Quintile'] += 1
  elif i < 0.6:
    d['3rd Quintile'] += 1
  elif i < 0.8:
    d['4th Quintile'] += 1
  elif i <= 1:
    d['5th Quintile'] += 1
  else:
    print('Error: Value outside expected range - ' + str(i))
  return d


def isfloat(num):
  if num is None:
    return False
  try:
      num = float(num)
      if math.isnan(num):
        return False
      return True
  except ValueError:
      return False
      

def make_quintile_dict(l):
  d = {'1st Quintile': 0, '2nd Quintile': 0, '3rd Quintile': 0, '4th Quintile': 0, '5th Quintile': 0, 'Missing': 0}
  l = list(l)
  for i in l:
    if isfloat(i):
      d = evaluate_quintile(d, i)
    else:
      d['Missing'] += 1
      c=+1
  print_dict(d, len(l))
  return d


def clean_pregravid_bmi(bmi):
  if isinstance(bmi, float):
    bmi = round(float(bmi), 1)
    if bmi >= 12 and bmi < 100:
      return bmi
  return None


def clean_bmi(bmi):
  if len(bmi) == 0:
    return None
  elif isinstance(bmi[0], float):
    bmi = round(float(bmi[0]), 1)
    if bmi >= 12 and bmi < 100:
      return bmi
  return None


def update_bmi(bmi, bmi_1):
  if isinstance(bmi, float):
    return bmi
  if isinstance(bmi_1, float):
    return bmi_1
  return bmi
      

def clean_up_bmi(df, df_flowsheets_1):
  for index, row in df.iterrows():
    pat_id = row['pat_id']
    delivery_date = row['ob_delivery_delivery_date']
    bmi = clean_pregravid_bmi(row['pregravid_bmi'])
    bmi_1 = clean_bmi(df_flowsheets_1.loc[(df_flowsheets_1['pat_id'] == pat_id) & (df_flowsheets_1['ob_delivery_delivery_date'] == delivery_date)]['BMI'].values)
    bmi_final = update_bmi(bmi, bmi_1)
    df.at[index, 'pregravid_bmi'] = bmi_final
  df['pregravid_bmi'] = pd.to_numeric(df['pregravid_bmi'])
  return df


# load cohorts
# load saved dataframes
df_cohort_maternity_vaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_mrna_expanded_6").drop('all_immunization_dates').toPandas().sort_index()
print('# Number of women vaccinated at time of delivery: ' + str(len(df_cohort_maternity_vaccinated)))

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_unvaccinated_expanded_6")
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom")
df_temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated")
df_temp.createOrReplaceTempView("temp")
df_temp = spark.sql("SELECT * FROM temp AS t WHERE '2021-01-26' <= t.ob_delivery_delivery_date")
df_temp.createOrReplaceTempView("temp")
df_add = spark.sql(
"""
FROM temp AS t
LEFT ANTI JOIN unvaccinated_mom AS um
ON um.pat_id = t.pat_id
AND um.instance = t.instance
AND um.episode_id = t.episode_id
AND um.child_episode_id = t.child_episode_id 
SELECT t.*
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id']).drop('all_immunization_dates').toPandas()
df_cohort_maternity_unvaccinated = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id']).drop('all_immunization_dates').toPandas().append(df_add).drop_duplicates(subset=['pat_id', 'episode_id', 'child_episode_id']).sort_index().fillna(0)
print('# Number of unvaccinated pregnant women who have delivered after 1-25-21: ' + str(len(df_cohort_maternity_unvaccinated)))


# evaluate parity
dict_cohort_maternity_vaccinated_parity = determine_parity(df_cohort_maternity_vaccinated)
dict_cohort_maternity_unvaccinated_parity = determine_parity(df_cohort_maternity_unvaccinated)
calc_fishers_exact_test(dict_cohort_maternity_vaccinated_parity, dict_cohort_maternity_unvaccinated_parity)


# evaluate gravidity
dict_cohort_maternity_vaccinated_gravidity = determine_gravidity(df_cohort_maternity_vaccinated)
dict_cohort_maternity_unvaccinated_gravidity = determine_gravidity(df_cohort_maternity_unvaccinated)
calc_fishers_exact_test(dict_cohort_maternity_vaccinated_gravidity, dict_cohort_maternity_unvaccinated_gravidity)


# evaluate preterm history
dict_cohort_maternity_vaccinated_preterm_history = determine_preterm_history(df_cohort_maternity_vaccinated)
dict_cohort_maternity_unvaccinated_preterm_history = determine_preterm_history(df_cohort_maternity_unvaccinated)
print(calc_fishers_exact_test(dict_cohort_maternity_vaccinated_preterm_history, dict_cohort_maternity_unvaccinated_preterm_history))


# evaluate race
dict_cohort_maternity_vaccinated_all_races = determine_race_distribution(df_cohort_maternity_vaccinated['race'])
dict_cohort_maternity_unvaccinated_all_races = determine_race_distribution(df_cohort_maternity_unvaccinated['race'])
print(calc_fishers_exact_test(dict_cohort_maternity_vaccinated_all_races, dict_cohort_maternity_unvaccinated_all_races))


# evaluate ethnicity
dict_cohort_maternity_vaccinated_ethnicity = determine_ethnic_distribution_2(df_cohort_maternity_vaccinated['ethnic_group'])
dict_cohort_maternity_unvaccinated_ethnicity = determine_ethnic_distribution_2(df_cohort_maternity_unvaccinated['ethnic_group'])
print(calc_fishers_exact_test(dict_cohort_maternity_vaccinated_ethnicity, dict_cohort_maternity_unvaccinated_ethnicity))


# evaluate age
make_histogram(df_cohort_maternity_vaccinated, 'age_at_start_dt', 1, 10, 60)
make_histogram(df_cohort_maternity_unvaccinated, 'age_at_start_dt', 1, 10, 60)
# run t-test on age at birth for covid-positive vs negative maternity cohorts
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'age_at_start_dt'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'age_at_start_dt'))
dict_vaccinated_age = make_age_dict(df_cohort_maternity_vaccinated)
dict_vaccinated_age.pop('Missing')
dict_unvaccinated_age = make_age_dict(df_cohort_maternity_unvaccinated)
dict_unvaccinated_age.pop('Missing')
print(calc_fishers_exact_test(dict_vaccinated_age, dict_unvaccinated_age))


# evaluate smoker status
calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'smoker')
vaccinated_smoker = df_cohort_maternity_vaccinated.smoker.sum()/len(df_cohort_maternity_vaccinated)
unvaccinated_smoker = df_cohort_maternity_unvaccinated.smoker.sum()/len(df_cohort_maternity_unvaccinated)
smoker_delta = vaccinated_smoker/unvaccinated_smoker
print('Percent of vaccinated maternity cohort that are smokers: ' + str(vaccinated_smoker))
print('Percent of unvaccinated cohort that are smokers: ' + str(unvaccinated_smoker))
print('Difference in percentage of smokers between the cohorts: ' + str(smoker_delta))


# evaluate illicit drug use
calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'illegal_drug_user')


# evaluate chronic diabetes status
calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'chronic_diabetes_with_pregnancy_status')


# evaluate chronic hypertension status
calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'chronic_hypertension_with_pregnancy_status')
vaccinated_hypertension = df_cohort_maternity_vaccinated.chronic_hypertension_with_pregnancy_status.sum()/len(df_cohort_maternity_vaccinated)
unvaccinated_hypertension = df_cohort_maternity_unvaccinated.chronic_hypertension_with_pregnancy_status.sum()/len(df_cohort_maternity_unvaccinated)
hypertension_delta = vaccinated_hypertension/unvaccinated_hypertension
print('Percent of vaccinated maternity cohort that have chronic hypertension: ' + str(vaccinated_hypertension))
print('Percent of unvaccinated maternity control cohort that have chronic hypertension: ' + str(unvaccinated_hypertension))
print('Difference in percentage of patients with chronic hypertension between the cohorts: ' + str(hypertension_delta))


# evaluate gestational diabetes
print(calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'gestational_diabetes_status'))


# evaluate gestational hypertension
print(calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'gestational_hypertension_status'))


# evaluate preeclampsia
print(calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'preeclampsia_status'))


# evaluate severe preeclampsia
print(calc_fishers_exact_test_2x2(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'severe_preeclampsia_status'))


# evaluate delivery method
dict_vaccinated_delivery_method = determine_count(df_cohort_maternity_vaccinated, 'delivery_delivery_method')
dict_unvaccinated_delivery_method = determine_count(df_cohort_maternity_unvaccinated, 'delivery_delivery_method')
dict_unvaccinated_delivery_method.pop(0)
calc_fishers_exact_test_for_delivery_method(dict_vaccinated_delivery_method, dict_unvaccinated_delivery_method)


# evaluate RUCA Categorization
dict_vaccinated_ruca_categorization = determine_count(df_cohort_maternity_vaccinated, 'ruca_categorization')
dict_unvaccinated_ruca_categorization = determine_count(df_cohort_maternity_unvaccinated, 'ruca_categorization')
dict_unvaccinated_ruca_categorization['Missing'] = dict_unvaccinated_ruca_categorization[0]
calc_fishers_exact_test(dict_vaccinated_ruca_categorization, dict_unvaccinated_ruca_categorization)


# reload unvaccinated cohort
df_cohort_maternity_unvaccinated = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id']).drop('all_immunization_dates').toPandas().append(df_add).drop_duplicates(subset=['pat_id', 'episode_id', 'child_episode_id']).sort_index()


# evaluate RPL themes
make_histogram(df_cohort_maternity_vaccinated, 'RPL_THEMES', .05, 0, 1)
make_histogram(df_cohort_maternity_unvaccinated, 'RPL_THEMES', .05, 0, 1)
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEMES'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEMES'))
dict_rpl_themes_unvaccinated = make_quintile_dict(df_cohort_maternity_unvaccinated['RPL_THEMES'])
dict_rpl_themes_vaccinated = make_quintile_dict(df_cohort_maternity_vaccinated['RPL_THEMES'])
calc_fishers_exact_test(dict_rpl_themes_vaccinated, dict_rpl_themes_unvaccinated)


# evaluate RPL theme 1: Socioeconomic Status
make_histogram(df_cohort_maternity_vaccinated, 'RPL_THEME1', .05, 0, 1)
make_histogram(df_cohort_maternity_unvaccinated, 'RPL_THEME1', .05, 0, 1)
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME1'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME1'))
dict_rpl_theme1_unvaccinated = make_quintile_dict(df_cohort_maternity_unvaccinated['RPL_THEME1'])
dict_rpl_theme1_vaccinated = make_quintile_dict(df_cohort_maternity_vaccinated['RPL_THEME1'])
calc_fishers_exact_test(dict_rpl_theme1_vaccinated, dict_rpl_theme1_unvaccinated)


# evaluate RPL theme 2: Household Composition and Disability
make_histogram(df_cohort_maternity_vaccinated, 'RPL_THEME2', .05, 0, 1)
make_histogram(df_cohort_maternity_unvaccinated, 'RPL_THEME2', .05, 0, 1)
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME2'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME2'))
dict_rpl_theme2_unvaccinated = make_quintile_dict(df_cohort_maternity_unvaccinated['RPL_THEME2'])
dict_rpl_theme2_vaccinated = make_quintile_dict(df_cohort_maternity_vaccinated['RPL_THEME2'])
calc_fishers_exact_test(dict_rpl_theme2_vaccinated, dict_rpl_theme2_unvaccinated)


# evaluate RPL theme 3: Language and Minority Status
make_histogram(df_cohort_maternity_vaccinated, 'RPL_THEME3', .05, 0, 1)
make_histogram(df_cohort_maternity_unvaccinated, 'RPL_THEME3', .05, 0, 1)
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME3'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME3'))
dict_rpl_theme3_unvaccinated = make_quintile_dict(df_cohort_maternity_unvaccinated['RPL_THEME3'])
dict_rpl_theme3_vaccinated = make_quintile_dict(df_cohort_maternity_vaccinated['RPL_THEME3'])
calc_fishers_exact_test(dict_rpl_theme3_vaccinated, dict_rpl_theme3_unvaccinated)


# evaluate RPL theme 4: Housing and transportation - housing density
make_histogram(df_cohort_maternity_vaccinated, 'RPL_THEME4', .05, 0, 1)
make_histogram(df_cohort_maternity_unvaccinated, 'RPL_THEME4', .05, 0, 1)
print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME4'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'RPL_THEME4'))
dict_rpl_theme4_unvaccinated = make_quintile_dict(df_cohort_maternity_unvaccinated['RPL_THEME4'])
dict_rpl_theme4_vaccinated = make_quintile_dict(df_cohort_maternity_vaccinated['RPL_THEME4'])
calc_fishers_exact_test(dict_rpl_theme4_vaccinated, dict_rpl_theme4_unvaccinated)


# evaluate pregravid BMI from delivery charts
df_cohort_maternity_vaccinated['pregravid_bmi'] = df_cohort_maternity_vaccinated['pregravid_bmi'].replace(0, np.nan)
print(len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum())
print((len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_vaccinated))
make_histogram(df_cohort_maternity_vaccinated, 'pregravid_bmi', 1, 12, 60)

df_cohort_maternity_unvaccinated['pregravid_bmi'] = df_cohort_maternity_unvaccinated['pregravid_bmi'].replace(0, np.nan)
print(len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum())
print((len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_unvaccinated))
make_histogram(df_cohort_maternity_unvaccinated, 'pregravid_bmi', 1, 12, 60)

print(run_t_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'pregravid_bmi'))
print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'pregravid_bmi'))


# for records missing pregravid BMI calculate BMI from height and weight from a visit within a year of pregnancy onset
vaccinated_mom = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna").select('pat_id', 'instance', 'ob_delivery_delivery_date', 'gestational_days') 
vaccinated_mom.createOrReplaceTempView("vaccinated_mom")
flo_dict = {
            'height':'(90011, 11)',
            }
flo_dict_form = {
                 'height':'/39.37 as height',
                }
for variable in flo_dict:
    tmp = spark.sql(
    """
    SELECT DISTINCT vm.pat_id, vm.instance, vm.ob_delivery_delivery_date, LAST(flowsheetentry.value)""" + flo_dict_form[variable] + """
    FROM ((vaccinated_mom as vm
      INNER JOIN rdp_phi.flowsheet ON vm.pat_id = rdp_phi.flowsheet.pat_id AND vm.instance = flowsheet.instance)
      INNER JOIN rdp_phi.flowsheetentry ON rdp_phi.flowsheet.FSD_ID = rdp_phi.flowsheetentry.FSD_ID)
    WHERE FLO_MEAS_ID in """ + flo_dict[variable] + """
      AND flowsheetentry.value IS NOT NULL
    GROUP BY vm.pat_id, vm.instance, vm.ob_delivery_delivery_date
    """
    )
vaccinated_mom = vaccinated_mom.join(tmp, ['pat_id', 'instance','ob_delivery_delivery_date'])
vaccinated_mom.createOrReplaceTempView("vaccinated_mom")

flo_dict = {
            'weight':'(90014, 14)',
            }
flo_dict_form = {
                 'weight':'/35.274 as weight',
                }
for variable in flo_dict:
    tmp = spark.sql(
    """
    SELECT DISTINCT vm.pat_id, vm.instance, vm.ob_delivery_delivery_date, LAST(flowsheetentry.value)""" + flo_dict_form[variable] + """
    FROM ((vaccinated_mom as vm
      INNER JOIN rdp_phi.flowsheet ON vm.pat_id = rdp_phi.flowsheet.pat_id AND vm.instance = flowsheet.instance)
      INNER JOIN rdp_phi.flowsheetentry ON rdp_phi.flowsheet.FSD_ID = rdp_phi.flowsheetentry.FSD_ID)
    WHERE flowsheetentry.RECORDED_TIME > date_sub(vm.ob_delivery_delivery_date - interval '12' month, vm.gestational_days)
      AND flowsheetentry.RECORDED_TIME <= date_sub(vm.ob_delivery_delivery_date + interval '3' month, vm.gestational_days)
      AND FLO_MEAS_ID in """ + flo_dict[variable] + """
      AND flowsheetentry.value IS NOT NULL
    GROUP BY vm.pat_id, vm.instance, vm.ob_delivery_delivery_date
    """
    )
vaccinated_mom = vaccinated_mom.join(tmp, ['pat_id', 'instance','ob_delivery_delivery_date'])
vaccinated_mom = vaccinated_mom.withColumn('patient_id', F.concat(F.col('instance'), F.col('pat_id')))\
           .drop(*['patient_id', 'instance'])\
           .withColumnRenamed('height', 'fs_height')\
           .withColumnRenamed('weight', 'fs_weight')\
           .withColumn('BMI', F.col('fs_weight')/F.col('fs_height')**2).dropDuplicates(['pat_id', 'ob_delivery_delivery_date']).toPandas()
print(len(vaccinated_mom))

unvaccinated_mom = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated").select('pat_id', 'instance', 'ob_delivery_delivery_date', 'gestational_days') 
unvaccinated_mom.createOrReplaceTempView("unvaccinated_mom")
flo_dict = {
            'height':'(90011, 11)',
            }
flo_dict_form = {
                 'height':'/39.37 as height',
                }
for variable in flo_dict:
    tmp = spark.sql(
    """
    SELECT DISTINCT vm.pat_id, vm.instance, vm.ob_delivery_delivery_date, LAST(flowsheetentry.value)""" + flo_dict_form[variable] + """
    FROM ((unvaccinated_mom as vm
      INNER JOIN rdp_phi.flowsheet ON vm.pat_id = rdp_phi.flowsheet.pat_id AND vm.instance = flowsheet.instance)
      INNER JOIN rdp_phi.flowsheetentry ON rdp_phi.flowsheet.FSD_ID = rdp_phi.flowsheetentry.FSD_ID)
    WHERE FLO_MEAS_ID in """ + flo_dict[variable] + """
      AND flowsheetentry.value IS NOT NULL
    GROUP BY vm.pat_id, vm.instance, vm.ob_delivery_delivery_date
    """
    )
unvaccinated_mom = unvaccinated_mom.join(tmp, ['pat_id', 'instance','ob_delivery_delivery_date'])
unvaccinated_mom.createOrReplaceTempView("unvaccinated_mom")

flo_dict = {
            'weight':'(90014, 14)',
            }
flo_dict_form = {
                 'weight':'/35.274 as weight',
                }
for variable in flo_dict:
    tmp = spark.sql(
    """
    SELECT DISTINCT vm.pat_id, vm.instance, vm.ob_delivery_delivery_date, LAST(flowsheetentry.value)""" + flo_dict_form[variable] + """
    FROM ((unvaccinated_mom as vm
      INNER JOIN rdp_phi.flowsheet ON vm.pat_id = rdp_phi.flowsheet.pat_id AND vm.instance = flowsheet.instance)
      INNER JOIN rdp_phi.flowsheetentry ON rdp_phi.flowsheet.FSD_ID = rdp_phi.flowsheetentry.FSD_ID)
    WHERE flowsheetentry.RECORDED_TIME > date_sub(vm.ob_delivery_delivery_date - interval '12' month, vm.gestational_days)
      AND flowsheetentry.RECORDED_TIME <= date_sub(vm.ob_delivery_delivery_date + interval '3' month, vm.gestational_days)
      AND FLO_MEAS_ID in """ + flo_dict[variable] + """
      AND flowsheetentry.value IS NOT NULL
    GROUP BY vm.pat_id, vm.instance, vm.ob_delivery_delivery_date
    """
    )
unvaccinated_mom = unvaccinated_mom.join(tmp, ['pat_id', 'instance','ob_delivery_delivery_date'])
unvaccinated_mom = unvaccinated_mom.withColumn('patient_id', F.concat(F.col('instance'), F.col('pat_id')))\
           .drop(*['patient_id', 'instance'])\
           .withColumnRenamed('height', 'fs_height')\
           .withColumnRenamed('weight', 'fs_weight')\
           .withColumn('BMI', F.col('fs_weight')/F.col('fs_height')**2).dropDuplicates(['pat_id', 'ob_delivery_delivery_date']).toPandas()
print(len(unvaccinated_mom))


# add new calculated pregravid BMI and save dataframes
df_cohort_maternity_vaccinated = clean_up_bmi(df_cohort_maternity_vaccinated, vaccinated_mom)
print('# Number of vaccinated pregnant people with pregravid BMI: ' + str(len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum()))
print('# Percentage of vaccinated pregnant people with pregravid BMI: ' + str(round(100*(len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_vaccinated), 1)) + '%')

df_cohort_maternity_unvaccinated = clean_up_bmi(df_cohort_maternity_unvaccinated, unvaccinated_mom)
print('# Number of unvaccinated pregnant people with pregravid BMI: ' + str(len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum()))
print('# Percentage of unvaccinated pregnant people with pregravid BMI: ' + str(round(100*(len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_unvaccinated), 1)) + '%')

vaccinated_temp_pyspark = spark.createDataFrame(df_cohort_maternity_vaccinated[["pat_id", "episode_id", "child_episode_id", "pregravid_bmi"]])
unvaccinated_temp_pyspark = spark.createDataFrame(df_cohort_maternity_unvaccinated[["pat_id", "episode_id", "child_episode_id", "pregravid_bmi"]])

# save updated pregravid bmi to table
write_data_frame_to_sandbox(vaccinated_temp_pyspark, 'snp2_cohort_maternity_vaccinated_mrna_expanded_6_pregravid_bmi', sandbox_db='rdp_phi_sandbox', replace=True)
write_data_frame_to_sandbox(unvaccinated_temp_pyspark, 'snp2_cohort_maternity_unvaccinated_expanded_6_pregravid_bmi', sandbox_db='rdp_phi_sandbox', replace=True)


# add pregravid bmi to cohort data tables and save
df_cohort_maternity_vaccinated_temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_mrna_expanded_6").drop('pregravid_bmi') 
df_cohort_maternity_vaccinated_temp.createOrReplaceTempView("vm")
vaccinated_temp_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna_expanded_6_pregravid_bmi")
vaccinated_temp_pyspark.createOrReplaceTempView("bmi")
df_cohort_maternity_vaccinated_bmi = spark.sql(
"""
SELECT /*+ RANGE_JOIN(c, 50) */ vm.*, bmi.pregravid_bmi
FROM vm
LEFT OUTER JOIN bmi
ON vm.pat_id = bmi.pat_id
  AND vm.episode_id = bmi.episode_id
  AND vm.child_episode_id = bmi.child_episode_id
""")

write_data_frame_to_sandbox(df_cohort_maternity_vaccinated_bmi, 'snp2_cohort_maternity_vaccinated_mrna_expanded_7', sandbox_db='rdp_phi_sandbox', replace=True)

df_unvaccinated_temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_unvaccinated_expanded_6")
df_unvaccinated_temp.createOrReplaceTempView("unvaccinated_mom")
df_temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated")
df_temp.createOrReplaceTempView("temp")
df_temp = spark.sql("SELECT * FROM temp AS t WHERE '2021-01-26' <= t.ob_delivery_delivery_date")
df_temp.createOrReplaceTempView("temp")
df_add = spark.sql(
"""
FROM temp AS t
LEFT ANTI JOIN unvaccinated_mom AS um
ON um.pat_id = t.pat_id
AND um.instance = t.instance
AND um.episode_id = t.episode_id
AND um.child_episode_id = t.child_episode_id 
SELECT t.*
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_maternity_unvaccinated_temp = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id']).unionByName(df_add, allowMissingColumns=True).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).drop('pregravid_bmi')
df_cohort_maternity_unvaccinated_temp.createOrReplaceTempView("uvm")


unvaccinated_temp_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_expanded_6_pregravid_bmi")
unvaccinated_temp_pyspark.createOrReplaceTempView("bmi")
df_cohort_maternity_unvaccinated_bmi = spark.sql(
"""
SELECT /*+ RANGE_JOIN(c, 50) */ uvm.*, bmi.pregravid_bmi
FROM uvm
LEFT OUTER JOIN bmi
ON uvm.pat_id = bmi.pat_id
  AND uvm.episode_id = bmi.episode_id
  AND uvm.child_episode_id = bmi.child_episode_id
""")


write_data_frame_to_sandbox(df_cohort_maternity_unvaccinated_bmi, 'snp2_cohort_maternity_unvaccinated_expanded_7', sandbox_db='rdp_phi_sandbox', replace=True)


# load saved dataframes
df_cohort_maternity_vaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna_expanded_7").toPandas()
print('# Number of women vaccinated at time of delivery: ' + str(len(df_cohort_maternity_vaccinated)))

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_expanded_7")
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom")
df_cohort_maternity_unvaccinated = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date").toPandas().sort_index()
print('# Number of unvaccinated pregnant women who have delivered after 1-25-21: ' + str(len(df_cohort_maternity_unvaccinated)))


# evaluate pregravid BMI
print('# Number of vaccinated patients with pregravid BMI:', str(len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum()))
print('# Percent of vaccinated patients with pregravid BMI:', str(round(100*(len(df_cohort_maternity_vaccinated) - df_cohort_maternity_vaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_vaccinated), 1)), '%')
make_histogram(df_cohort_maternity_vaccinated, 'pregravid_bmi', 1, 12, 60)

print('# Number of unvaccinated patients with pregravid BMI:', str(len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum()))
print('# Percent of unvaccinated patients with pregravid BMI:', str(round(100*(len(df_cohort_maternity_unvaccinated) - df_cohort_maternity_unvaccinated['pregravid_bmi'].isna().sum())/len(df_cohort_maternity_unvaccinated), 1)), '%')
make_histogram(df_cohort_maternity_unvaccinated, 'pregravid_bmi', 1, 12, 60)

print(run_mann_whitney_u_test(df_cohort_maternity_vaccinated, df_cohort_maternity_unvaccinated, 'pregravid_bmi'))

dict_vaccinated_bmi = make_bmi_dict(df_cohort_maternity_vaccinated)
dict_unvaccinated_bmi = make_bmi_dict(df_cohort_maternity_unvaccinated)
print(calc_fishers_exact_test(dict_vaccinated_bmi, dict_unvaccinated_bmi))


# evaluate insurance status
dict_unvaccinated_insurance = determine_insurance_distribution(df_cohort_maternity_unvaccinated)
dict_unvaccinated_insurance = {'Medicaid/Uninsured': dict_unvaccinated_insurance['Medicaid'] + dict_unvaccinated_insurance['Medicare'] + dict_unvaccinated_insurance['Uninsured-Self-Pay'], 'Commercial': dict_unvaccinated_insurance['Commercial']}
dict_vaccinated_insurance = determine_insurance_distribution(df_cohort_maternity_vaccinated)
dict_vaccinated_insurance = {'Medicaid/Uninsured': dict_vaccinated_insurance['Medicaid'] + dict_vaccinated_insurance['Medicare'] + dict_vaccinated_insurance['Uninsured-Self-Pay'], 'Commercial': dict_vaccinated_insurance['Commercial']}
print(calc_fishers_exact_test(dict_vaccinated_insurance, dict_unvaccinated_insurance))
medicaid_delta = (dict_vaccinated_insurance['Medicaid/Uninsured']/len(dict_vaccinated_insurance))/(dict_unvaccinated_insurance['Medicaid/Uninsured']/len(dict_unvaccinated_insurance))
print('Difference in percentage of patients on Medicaid/Uninsured: ' + str(medicaid_delta))


# evaluate fetal sex
dict_fetal_sex_unvaccinated = determine_fetal_sex(df_cohort_maternity_unvaccinated)
dict_fetal_sex_vaccinated = determine_fetal_sex(df_cohort_maternity_vaccinated)
print(calc_fishers_exact_test(dict_fetal_sex_unvaccinated, dict_fetal_sex_vaccinated))
# calculate difference in male / female fetal ratio between what occurred and what was expected
(dict_fetal_sex_vaccinated['Male']/dict_fetal_sex_vaccinated['Female'])/(dict_fetal_sex_unvaccinated['Male']/dict_fetal_sex_unvaccinated['Female'])


# print out summary statistics of cohort demographics
print('Number of Vaccinated Moms: ' + str(len(df_cohort_maternity_vaccinated)))
print('Number of Unvaccinated Moms: ' + str(len(df_cohort_maternity_unvaccinated)))
print('\n')
print('Median Age at Birth Vaccinated: ' + str(df_cohort_maternity_vaccinated.age_at_start_dt.median()))
print('IQR Age at Birth Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.age_at_start_dt)))
print('\n')
print('Median Age at Birth Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.age_at_start_dt.median()))
print('IQR Age at Birth Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.age_at_start_dt)))
print('\n')
print('Median Pre-Pregnancy BMI Vaccinated: ' + str(df_cohort_maternity_vaccinated.pregravid_bmi.dropna().median()))
print('IQR Pre-Pregnancy BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.pregravid_bmi.dropna().astype(float))))
print('\n')
print('Medican Pre-Pregnancy BMI Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.pregravid_bmi.dropna().median()))
print('IQR Pre-Pregnancy BMI Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.pregravid_bmi.dropna().astype(float))))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_vaccinated.Parity))]
print('Median Parity Vaccinated: ' + str(statistics.median(temp)))
print('IQR Parity Vaccinated: ' + str(stats.iqr(temp)))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_unvaccinated.Parity))]
print('Median Parity Unvaccianted: ' + str(statistics.median(temp)))
print('IQR Parity Unvaccinated: ' + str(stats.iqr(temp)))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_vaccinated.Gravidity))]
print('Median Gravidity Vaccinated: ' +  str(statistics.median(temp)))
print('IQR Gravidity Vaccinated: ' + str(stats.iqr(temp)))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_unvaccinated.Gravidity))]
print('Median Gravidity Unvaccinated: ' + str(statistics.median(temp)))
print('IQR Gravidity Unvaccinated: ' + str(stats.iqr(temp)))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_vaccinated.Preterm_history))]
print('Median Preterm History Vaccinated: ' + str(statistics.median(temp)))
print('IQR Preterm History Vaccinated: ' + str(stats.iqr(temp)))
print('\n')
temp = [float(i) for i in list(filter(None, df_cohort_maternity_unvaccinated.Preterm_history))]
print('Median Preterm History Unvaccinated: ' + str(statistics.median(temp)))
print('IQR Preterm History Unvaccinated: ' + str(stats.iqr(temp)))
print('\n')
print('Median RPL Theme Summary Vaccinated: ' + str(df_cohort_maternity_vaccinated.RPL_THEMES[df_cohort_maternity_vaccinated.RPL_THEMES != "-999.0"].dropna().median()))
print('IQR RPL Theme Summary BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.RPL_THEMES[df_cohort_maternity_vaccinated.RPL_THEMES != "-999.0"].dropna().astype(float))))
print('\n')
print('Medican RPL Theme Summary Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.RPL_THEMES[df_cohort_maternity_unvaccinated.RPL_THEMES != "-999.0"].dropna().median()))
print('IQR RPL Theme Summary Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.RPL_THEMES[df_cohort_maternity_unvaccinated.RPL_THEMES != "-999.0"].dropna().astype(float))))
print('\n')
print('Median RPL Theme 1 Vaccinated: ' + str(df_cohort_maternity_vaccinated.RPL_THEME1[df_cohort_maternity_vaccinated.RPL_THEME1 != "-999.0"].dropna().median()))
print('IQR RPL Theme 1 BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.RPL_THEME1[df_cohort_maternity_vaccinated.RPL_THEME1 != "-999.0"].dropna().astype(float))))
print('\n')
print('Medican RPL Theme 1 Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.RPL_THEME1[df_cohort_maternity_unvaccinated.RPL_THEME1 != "-999.0"].dropna().median()))
print('IQR RPL Theme 1 Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.RPL_THEME1[df_cohort_maternity_unvaccinated.RPL_THEME1 != "-999.0"].dropna().astype(float))))
print('\n')
print('Median RPL Theme 2 Vaccinated: ' + str(df_cohort_maternity_vaccinated.RPL_THEME2[df_cohort_maternity_vaccinated.RPL_THEME2 != "-999.0"].dropna().median()))
print('IQR RPL Theme 2 BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.RPL_THEME2[df_cohort_maternity_vaccinated.RPL_THEME2 != "-999.0"].dropna().astype(float))))
print('\n')
print('Medican RPL Theme 2 Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.RPL_THEME2[df_cohort_maternity_unvaccinated.RPL_THEME2 != "-999.0"].dropna().median()))
print('IQR RPL Theme 2 Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.RPL_THEME2[df_cohort_maternity_unvaccinated.RPL_THEME2 != "-999.0"].dropna().astype(float))))
print('\n')
print('Median RPL Theme 3 Vaccinated: ' + str(df_cohort_maternity_vaccinated.RPL_THEME3[df_cohort_maternity_vaccinated.RPL_THEME3 != "-999.0"].dropna().median()))
print('IQR RPL Theme 3 BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.RPL_THEME3[df_cohort_maternity_vaccinated.RPL_THEME4 != "-999.0"].dropna().astype(float))))
print('\n')
print('Medican RPL Theme 3 Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.RPL_THEME3[df_cohort_maternity_unvaccinated.RPL_THEME3 != "-999.0"].dropna().median()))
print('IQR RPL Theme 3 Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.RPL_THEME3[df_cohort_maternity_unvaccinated.RPL_THEME3 != "-999.0"].dropna().astype(float))))
print('\n')
print('Median RPL Theme 4 Vaccinated: ' + str(df_cohort_maternity_vaccinated.RPL_THEME4[df_cohort_maternity_vaccinated.RPL_THEME4 != "-999.0"].dropna().median()))
print('IQR RPL Theme 4 BMI Vaccinated: ' + str(stats.iqr(df_cohort_maternity_vaccinated.RPL_THEME4[df_cohort_maternity_vaccinated.RPL_THEME4 != "-999.0"].dropna().astype(float))))
print('\n')
print('Medican RPL Theme 4 Unvaccinated: ' + str(df_cohort_maternity_unvaccinated.RPL_THEME4[df_cohort_maternity_unvaccinated.RPL_THEME4 != "-999.0"].dropna().median()))
print('IQR RPL Theme 4 Unvaccinated: ' + str(stats.iqr(df_cohort_maternity_unvaccinated.RPL_THEME4[df_cohort_maternity_unvaccinated.RPL_THEME4 != "-999.0"].dropna().astype(float))))
print('\n')
print('Number of Pre-Delta COVID-Positive Patients with Chronic Diabetes: ' + str(df_cohort_maternity_vaccinated.chronic_diabetes_with_pregnancy_status.sum()))
print('Number of Pre-Delta COVID-Positive Patients with Chronic Hypertension: ' + str(df_cohort_maternity_vaccinated.chronic_hypertension_with_pregnancy_status.sum()))
print('Number of Pre-Delta COVID-Positive Patients with Gestational Diabetes: ' + str(df_cohort_maternity_vaccinated.gestational_diabetes_status.sum()))
print('Number of Pre-Delta COVID-Positive Patients with Gestational Hypertension: ' + str(df_cohort_maternity_vaccinated.gestational_hypertension_status.sum()))
print('Number of Pre-Delta COVID-Positive Patients with Preeclampsia: ' + str(df_cohort_maternity_vaccinated.preeclampsia_status.sum()))
print('Number of Pre-Delta COVID-Positive Patients with Severe Preeclampsia: ' + str(df_cohort_maternity_vaccinated.severe_preeclampsia_status.sum()))
print('\n')
print('Number of Delta COVID-Positive Patients with Chronic Diabetes: ' + str(df_cohort_maternity_unvaccinated.chronic_diabetes_with_pregnancy_status.sum()))
print('Number of Delta COVID-Positive Patients with Chronic Hypertension: ' + str(df_cohort_maternity_unvaccinated.chronic_hypertension_with_pregnancy_status.sum()))
print('Number of Delta COVID-Positive Patients with Gestational Diabetes: ' + str(df_cohort_maternity_unvaccinated.gestational_diabetes_status.sum()))
print('Number of Delta COVID-Positive Patients with Gestational Hypertension: ' + str(df_cohort_maternity_unvaccinated.gestational_hypertension_status.sum()))
print('Number of Delta COVID-Positive Patients with Preeclampsia: ' + str(df_cohort_maternity_unvaccinated.preeclampsia_status.sum()))
print('Number of Delta COVID-Positive Patients with Severe Preeclampsia: ' + str(df_cohort_maternity_unvaccinated.severe_preeclampsia_status.sum()))
print('\n')
print('Number of Vaccianted Patients that are Smokers: ' + str(df_cohort_maternity_vaccinated.smoker.sum()))
print('Number of Vaccinated Patients that are Illegal Drug Users: ' + str(df_cohort_maternity_vaccinated.illegal_drug_user.sum()))
print('\n')
print('Number of Unvaccinated Patients that are Smokers: ' + str(df_cohort_maternity_unvaccinated.smoker.sum()))
print('Number of Unvaccinated Patients that are Illegal Drug Users: ' + str(df_cohort_maternity_unvaccinated.illegal_drug_user.sum()))
print('\n')
print('Number of Vaccinated Patients Baby Fetal Sex is Female: ' + str(dict_fetal_sex_vaccinated['Female']))
print('Number of Vaccinated Patients Baby Fetal Sex is Male: ' + str(dict_fetal_sex_vaccinated['Male']))
print('\n')
print('Number of Unvaccinated Patients Baby Fetal Sex is Female: ' + str(dict_fetal_sex_unvaccinated['Female']))
print('Number of Unvaccinated Patients Baby Fetal Sex is Male: ' + str(dict_fetal_sex_unvaccinated['Male']))
print('\n')


# check distribution of the data
# maternal age
print('Maternal Age Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.age_at_start_dt, line='s')
plt.show()

print('Maternal Age Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.age_at_start_dt, line='s')
plt.show()
print('\n')


# pregravid bmi
print('Pregravid BMI Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.pregravid_bmi.dropna().astype(float), line='s')
plt.show()

print('Pregravid BMI Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.pregravid_bmi.dropna().astype(float), line='s')
plt.show()
print('\n')


# RPL_THEMES
print('RPL Themes Summary Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.RPL_THEMES[df_cohort_maternity_vaccinated.RPL_THEMES != "-999.0"].dropna().astype(float), line='s')
plt.show()

print('RPL Themes Summary Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.RPL_THEMES[df_cohort_maternity_unvaccinated.RPL_THEMES != "-999.0"].dropna().astype(float), line='s')
plt.show()
print('\n')


# RPL_THEME1
print('RPL Theme 1 Summary Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.RPL_THEME1[df_cohort_maternity_vaccinated.RPL_THEME1 != "-999.0"].dropna().astype(float), line='s')
plt.show()

print('RPL Theme 1 Summary Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.RPL_THEME1[df_cohort_maternity_unvaccinated.RPL_THEME1 != "-999.0"].dropna().astype(float), line='s')
plt.show()
print('\n')


# RPL_THEME2
print('RPL Theme 2 Summary Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.RPL_THEME2[df_cohort_maternity_vaccinated.RPL_THEME2 != "-999.0"].dropna().astype(float), line='s')
plt.show()

print('RPL Theme 2 Summary Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.RPL_THEME2[df_cohort_maternity_unvaccinated.RPL_THEME2 != "-999.0"].dropna().astype(float), line='s')
plt.show()
print('\n')


# RPL_THEME3
print('RPL Theme 3 Summary Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.RPL_THEME3[df_cohort_maternity_vaccinated.RPL_THEME3 != "-999.0"].dropna().astype(float), line='s')
plt.show()

print('RPL Theme 3 Summary Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.RPL_THEME3[df_cohort_maternity_unvaccinated.RPL_THEME3 != "-999.0"].dropna().astype(float), line='s')
plt.show()
print('\n')


# RPL_THEME4
print('RPL Theme 4 Summary Vaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_vaccinated.RPL_THEME4[df_cohort_maternity_vaccinated.RPL_THEME4 != "-999.0"].dropna().astype(float), line='s')
plt.show()

print('RPL Theme 4 Summary Unvaccinated Quantile Quantile Plot')
qqplot(df_cohort_maternity_unvaccinated.RPL_THEME4[df_cohort_maternity_unvaccinated.RPL_THEME4 != "-999.0"].dropna().astype(float), line='s')
plt.show()
print('\n')

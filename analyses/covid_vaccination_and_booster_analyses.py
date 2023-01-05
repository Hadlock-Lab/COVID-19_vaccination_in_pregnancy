# Author: Samantha Piekos
# Date: 11/1/22


# load environment
import datetime
import numpy as np
import pandas as pd
import seaborn
import statistics

from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed
from pyspark.sql.functions import col, lit, unix_timestamp
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

dbutils.library.installPyPI("rpy2")
import rpy2

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42
seed(31415)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions
import COVID19_vaccination_in_pregnancy.utilities.calculate_fetal_growth_percentile.py


# define function
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
  print(dict_preterm_status)
  return(dict_preterm_status)


def determine_term_or_preterm_status(df):
  d = {'term': 0, 'preterm': 0}
  for index, row in df.iterrows():
    gestational_days = row['gestational_days']
    if gestational_days >= 259:
      d['term'] += 1
    else:
      d['preterm'] += 1
  print(d)
  return(d)


def count_trimester_of_infection(l):
  l = list(l)
  d = {'1st trimester': 0, '2nd trimester': 0, '3rd trimester': 0}
  for i in l:
    d[i] += 1
  print(d)
  return d


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
  
  
def calc_fishers_exact_test_2x2(dict_obs, dict_exp):
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
  
  
def run_t_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.ttest_ind(data_1, data_2))


def run_mann_whitney_u_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.mannwhitneyu(data_1, data_2))


def add_to_fetal_living_status_dict(d, k):
  if k is None:
    d['None'] += 1
  elif k == 'Living':
    d['Living'] += 1
  else:
    d['Fetal Demise'] += 1
  return(d)


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
  print(d)
  return(d)


def make_low_birth_weight_dict(df):
  d = {'low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 88.1849:
      d['low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print(d)
  return d


def make_very_low_birth_weight_dict(df):
  d = {'very_low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 52.91094:
      d['very_low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print(d)
  return d


def make_stillbirth_dict(df):
  d = {'Living': 0, 'Fetal Demise': 0}
  for index, row in df.iterrows():
    if row['ob_history_last_known_living_status'] == 'Fetal Demise':
      d['Fetal Demise'] += 1
    else:
      d['Living'] += 1  # reported as 'Living' or missing field
  print(d)
  return d


def make_data_list(list_of_dicts, standardize):
  data = []
  for d in list_of_dicts:
    l = []
    for i in d.values():
      if standardize:
        i = i/sum(d.values())
      l.append(i)
    data.append(l)
  return(data)


def format_x_labels(labels):
  new_labels = []
  for l in labels:
    temp = l.split('_')
    name = ''
    for i in temp:
      name = name + ' ' + i.capitalize()
    new_labels.append(name[1:])
  return(new_labels)


def make_bar_chart(list_of_dicts, dict_color, standardize=True):
  group_labels = list(dict_color.keys())
  n = int(len(list_of_dicts[0]))
  data = []
  labels = format_x_labels(list_of_dicts[0].keys())
  c = 0
  data = make_data_list(list_of_dicts, standardize)
  x = np.arange(n)
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  if standardize:
    ax.set_ylabel('Percentage of Preterm Births')
    ax.set_ylim(0,1)
  else:
    ax.set_ylabel('Number of Preterm Births')
  while c < len(list_of_dicts):
    ax.bar(x + 1*c/n, data[c], color=dict_color[group_labels[c]], width = 1/n, label = group_labels[c])
    c += 1
  ax.legend()
  display(plt.show())


# load cohorts

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
  print(dict_preterm_status)
  return(dict_preterm_status)


def determine_term_or_preterm_status(df):
  d = {'term': 0, 'preterm': 0}
  for index, row in df.iterrows():
    gestational_days = row['gestational_days']
    if gestational_days >= 259:
      d['term'] += 1
    else:
      d['preterm'] += 1
  print(d)
  return(d)


def count_trimester_of_infection(l):
  l = list(l)
  d = {'1st trimester': 0, '2nd trimester': 0, '3rd trimester': 0}
  for i in l:
    d[i] += 1
  print(d)
  return d


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
  
  
def calc_fishers_exact_test_2x2(dict_obs, dict_exp):
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
  
  
def run_t_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.ttest_ind(data_1, data_2))


def run_mann_whitney_u_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.mannwhitneyu(data_1, data_2))


def add_to_fetal_living_status_dict(d, k):
  if k is None:
    d['None'] += 1
  elif k == 'Living':
    d['Living'] += 1
  else:
    d['Fetal Demise'] += 1
  return(d)


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
  print(d)
  return(d)


def make_low_birth_weight_dict(df):
  d = {'low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 88.1849:
      d['low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print(d)
  return d


def make_very_low_birth_weight_dict(df):
  d = {'very_low_birth_weight': 0, 'normal_birth_weight': 0}
  for index, row in df.dropna(subset=['delivery_infant_birth_weight_oz']).iterrows():
    weight = row['delivery_infant_birth_weight_oz']
    if weight < 52.91094:
      d['very_low_birth_weight'] += 1
    else:
      d['normal_birth_weight'] += 1
  print(d)
  return d


def make_stillbirth_dict(df):
  d = {'Living': 0, 'Fetal Demise': 0}
  for index, row in df.iterrows():
    if row['ob_history_last_known_living_status'] == 'Fetal Demise':
      d['Fetal Demise'] += 1
    else:
      d['Living'] += 1  # reported as 'Living' or missing field
  print(d)
  return d


def make_data_list(list_of_dicts, standardize):
  data = []
  for d in list_of_dicts:
    l = []
    for i in d.values():
      if standardize:
        i = i/sum(d.values())
      l.append(i)
    data.append(l)
  return(data)


def format_x_labels(labels):
  new_labels = []
  for l in labels:
    temp = l.split('_')
    name = ''
    for i in temp:
      name = name + ' ' + i.capitalize()
    new_labels.append(name[1:])
  return(new_labels)


def make_bar_chart(list_of_dicts, dict_color, standardize=True):
  group_labels = list(dict_color.keys())
  n = int(len(list_of_dicts[0]))
  data = []
  labels = format_x_labels(list_of_dicts[0].keys())
  c = 0
  data = make_data_list(list_of_dicts, standardize)
  x = np.arange(n)
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  if standardize:
    ax.set_ylabel('Percentage of Preterm Births')
    ax.set_ylim(0,1)
  else:
    ax.set_ylabel('Number of Preterm Births')
  while c < len(list_of_dicts):
    ax.bar(x + 1*c/n, data[c], color=dict_color[group_labels[c]], width = 1/n, label = group_labels[c])
    c += 1
  ax.legend()
  display(plt.show())


def plot_histogram(data):
  n_bins = 40
  x_min, x_max = 0, 450
  
  plt.hist(data, bins=n_bins, range=(x_min, x_max), color="c")
  plt.axvline(x=182, color='k', linestyle='--')
  
  plt.xlim(x_min, x_max)
  plt.xlabel('Number of Days from Full Vaccination Status')
  plt.ylabel('Number of People\nReceiving Booster Shots')
  plt.tight_layout()
  plt.show()


# compare preterm birth rates between cohorts
dict_vaccinated_but_not_boosted_preterm_status = determine_preterm_status(df_vaccinated_but_not_boosted)
dict_vaccinated_but_not_boosted_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_but_not_boosted)
conversion = dict_vaccinated_but_not_boosted_term_or_preterm_status['term']/sum(dict_vaccinated_but_not_boosted_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_term_or_preterm_status['preterm'], dict_vaccinated_but_not_boosted_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_term_or_preterm_status['preterm']/(dict_vaccinated_but_not_boosted_term_or_preterm_status['term'] + dict_vaccinated_but_not_boosted_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_boosted_preterm_status = determine_preterm_status(df_boosted)
dict_boosted_term_or_preterm_status = determine_term_or_preterm_status(df_boosted)
conversion = dict_boosted_term_or_preterm_status['term']/sum(dict_boosted_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_boosted_term_or_preterm_status['preterm'], dict_boosted_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_term_or_preterm_status['preterm']/(dict_boosted_term_or_preterm_status['term'] + dict_boosted_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_boosted_preterm_status, dict_vaccinated_but_not_boosted_preterm_status))
dict_boosted_preterm_status.pop('term')
dict_vaccinated_but_not_boosted_preterm_status.pop('term')
print(calc_fishers_exact_test(dict_boosted_preterm_status, dict_vaccinated_but_not_boosted_preterm_status))
print(calc_fishers_exact_test(dict_boosted_term_or_preterm_status, dict_vaccinated_but_not_boosted_term_or_preterm_status))

dict_vaccinated_but_not_boosted_matched_preterm_status = determine_preterm_status(df_vaccinated_but_not_boosted_matched)
dict_vaccinated_but_not_boosted_matched_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_but_not_boosted_matched)
conversion = dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['term']/sum(dict_vaccinated_but_not_boosted_matched_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['preterm'], dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['preterm']/(dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['term'] + dict_vaccinated_but_not_boosted_matched_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_boosted_preterm_status, dict_vaccinated_but_not_boosted_matched_preterm_status))
dict_vaccinated_but_not_boosted_matched_preterm_status.pop('term')
print(calc_fishers_exact_test(dict_boosted_preterm_status, dict_vaccinated_but_not_boosted_matched_preterm_status))
print(calc_fishers_exact_test(dict_boosted_term_or_preterm_status, dict_vaccinated_but_not_boosted_matched_term_or_preterm_status))

make_bar_chart([dict_boosted_preterm_status, dict_vaccinated_but_not_boosted_matched_preterm_status, dict_vaccinated_but_not_boosted_preterm_status], {'Boosted': 'c', 'Vaccinated, But Not Boosted Matched': 'lightpink', 'Vaccinated, But Not Boosted': 'goldenrod'}, standardize=True)


# evaluate gestational days at delivery
temp = [i for i in df_vaccinated_but_not_boosted.gestational_days if str(i) != 'nan']
print('Median Vaccinated, but not Boosted Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Vaccinated, but not Boosted Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

temp = [i for i in df_boosted.gestational_days if str(i) != 'nan']
print('Median Boosted Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Boosted Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_boosted, df_vaccinated_but_not_boosted, 'gestational_days'))

temp = [i for i in df_vaccinated_but_not_boosted_matched.gestational_days if str(i) != 'nan']
print('Median Vaccinated, but not Boosted Matched Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Vaccinated, but not Boosted Matched Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_boosted, df_vaccinated_but_not_boosted_matched, 'gestational_days'))


# evalute birth weight
print('Median Vaccinated, but not Boosted Baby Birth Weight: ' + str(df_vaccinated_but_not_boosted.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated, but not Boosted Baby Birth Weight: ' + str(stats.iqr(df_vaccinated_but_not_boosted.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print('Median Boosted Patients Baby Birth Weight: ' + str(df_boosted.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Boosted Patients Baby Birth Weight: ' + str(stats.iqr(df_boosted.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_boosted, df_vaccinated_but_not_boosted, 'delivery_infant_birth_weight_oz'))

print('Median Vaccinated, but not Boosted Matched Baby Birth Weight: ' + str(df_vaccinated_but_not_boosted_matched.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated, but not Boosted Matched Baby Birth Weight: ' + str(stats.iqr(df_vaccinated_but_not_boosted_matched.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_boosted, df_vaccinated_but_not_boosted_matched, 'delivery_infant_birth_weight_oz'))


# evaluate fetal growth percentiles and small for gestational age
vaccinated_but_not_boosted_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_but_not_boosted)
print('Median Vaccinated, but not Boosted Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_but_not_boosted_fetal_growth_percentiles)))
print('IQR Vaccinated, but not Boosted Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_but_not_boosted_fetal_growth_percentiles)))
print('\n')

boosted_fetal_growth_percentiles = get_fetal_growth_percentiles(df_boosted)
print('Median Boosted Patients Baby Fetal Growth Percentile: ' + str(median(boosted_fetal_growth_percentiles)))
print('IQR Boosted Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(boosted_fetal_growth_percentiles)))
print('\n')
print(stats.mannwhitneyu(boosted_fetal_growth_percentiles, vaccinated_but_not_boosted_fetal_growth_percentiles))
print('\n')

dict_vaccinated_but_not_boosted_sga = count_small_for_gestational_age(vaccinated_but_not_boosted_fetal_growth_percentiles)
conversion = dict_vaccinated_but_not_boosted_sga['normal']/sum(dict_vaccinated_but_not_boosted_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_sga['SGA'],dict_vaccinated_but_not_boosted_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_sga['SGA']/(dict_vaccinated_but_not_boosted_sga['normal'] + dict_vaccinated_but_not_boosted_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_boosted_sga = count_small_for_gestational_age(boosted_fetal_growth_percentiles)
conversion = dict_boosted_sga['normal']/sum(dict_boosted_sga.values())
ci_low, ci_up = proportion_confint(dict_boosted_sga['SGA'],dict_boosted_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_sga['SGA']/(dict_boosted_sga['normal'] + dict_boosted_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_boosted_sga, dict_vaccinated_but_not_boosted_sga))


vaccinated_but_not_boosted_matched_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_but_not_boosted_matched)
print('Median Vaccinated, but not Boosted Matched Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_but_not_boosted_matched_fetal_growth_percentiles)))
print('IQR Vaccinated, but not Boosted Matched Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_but_not_boosted_matched_fetal_growth_percentiles)))
print('\n')

print(stats.mannwhitneyu(boosted_fetal_growth_percentiles, vaccinated_but_not_boosted_matched_fetal_growth_percentiles))
print('\n')

dict_vaccinated_but_not_boosted_matched_sga = count_small_for_gestational_age(vaccinated_but_not_boosted_matched_fetal_growth_percentiles)
conversion = dict_vaccinated_but_not_boosted_matched_sga['normal']/sum(dict_vaccinated_but_not_boosted_matched_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_sga['SGA'],dict_vaccinated_but_not_boosted_matched_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_sga['SGA']/(dict_vaccinated_but_not_boosted_matched_sga['normal'] + dict_vaccinated_but_not_boosted_matched_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_boosted_sga, dict_vaccinated_but_not_boosted_matched_sga))


# evaluate stillbirth rates
dict_vaccinated_but_not_boosted_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_but_not_boosted.iterrows():
  dict_vaccinated_but_not_boosted_fetal_death = add_to_fetal_living_status_dict(dict_vaccinated_but_not_boosted_fetal_death, row['ob_history_last_known_living_status'])
  if row['ob_history_last_known_living_status'] != 'Living' and row['ob_history_last_known_living_status'] is not None:  # check gender of deceased fetus
    print(row['ob_hx_infant_sex'])
print(dict_vaccinated_but_not_boosted_fetal_death)
dict_vaccinated_but_not_boosted_stillbirth = make_stillbirth_dict(df_vaccinated_but_not_boosted)
conversion = dict_vaccinated_but_not_boosted_stillbirth['Living']/sum(dict_vaccinated_but_not_boosted_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_stillbirth['Fetal Demise'], dict_vaccinated_but_not_boosted_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_stillbirth['Fetal Demise']/(dict_vaccinated_but_not_boosted_stillbirth['Living'] + dict_vaccinated_but_not_boosted_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_boosted_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_boosted.iterrows():
  dict_boosted_fetal_death = add_to_fetal_living_status_dict(dict_boosted_fetal_death, row['ob_history_last_known_living_status'])
  if row['ob_history_last_known_living_status'] != 'Living' and row['ob_history_last_known_living_status'] is not None:  # check gender of deceased fetus
    print(row['ob_hx_infant_sex'])
print(dict_boosted_fetal_death)
dict_boosted_stillbirth = make_stillbirth_dict(df_boosted)
conversion = dict_boosted_stillbirth['Living']/sum(dict_boosted_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_boosted_stillbirth['Fetal Demise'], dict_boosted_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_stillbirth['Fetal Demise']/(dict_boosted_stillbirth['Living'] + dict_boosted_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_boosted_fetal_death, dict_vaccinated_but_not_boosted_fetal_death))
print(calc_fishers_exact_test_not_simulated(make_stillbirth_dict(df_boosted), make_stillbirth_dict(df_vaccinated_but_not_boosted)))

dict_vaccinated_but_not_boosted_matched_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_but_not_boosted_matched.iterrows():
  dict_vaccinated_but_not_boosted_matched_fetal_death = add_to_fetal_living_status_dict(dict_vaccinated_but_not_boosted_matched_fetal_death, row['ob_history_last_known_living_status'])
  if row['ob_history_last_known_living_status'] != 'Living' and row['ob_history_last_known_living_status'] is not None:  # check gender of deceased fetus
    print(row['ob_hx_infant_sex'])
print(dict_vaccinated_but_not_boosted_matched_fetal_death)
dict_vaccinated_but_not_boosted_matched_stillbirth = make_stillbirth_dict(df_vaccinated_but_not_boosted_matched)
conversion = dict_vaccinated_but_not_boosted_matched_stillbirth['Living']/sum(dict_vaccinated_but_not_boosted_matched_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_stillbirth['Fetal Demise'], dict_vaccinated_but_not_boosted_matched_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_stillbirth['Fetal Demise']/(dict_vaccinated_but_not_boosted_matched_stillbirth['Living'] + dict_vaccinated_but_not_boosted_matched_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_boosted_fetal_death, dict_vaccinated_but_not_boosted_matched_fetal_death))
print(calc_fishers_exact_test_not_simulated(make_stillbirth_dict(df_boosted), make_stillbirth_dict(df_vaccinated_but_not_boosted_matched)))


# evalute low birth weight rates
dict_vaccinated_but_not_boosted_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_but_not_boosted)
conversion = dict_vaccinated_but_not_boosted_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_but_not_boosted_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_low_birth_weight['low_birth_weight'], dict_vaccinated_but_not_boosted_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_low_birth_weight['low_birth_weight']/(dict_vaccinated_but_not_boosted_low_birth_weight['normal_birth_weight'] + dict_vaccinated_but_not_boosted_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_boosted_low_birth_weight = make_low_birth_weight_dict(df_boosted)
conversion = dict_boosted_low_birth_weight['normal_birth_weight']/sum(dict_boosted_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_boosted_low_birth_weight['low_birth_weight'], dict_boosted_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_low_birth_weight['low_birth_weight']/(dict_boosted_low_birth_weight['normal_birth_weight'] + dict_boosted_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_boosted_low_birth_weight, dict_vaccinated_but_not_boosted_low_birth_weight)

dict_vaccinated_but_not_boosted_matched_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_but_not_boosted_matched)
conversion = dict_vaccinated_but_not_boosted_matched_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_but_not_boosted_matched_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_low_birth_weight['low_birth_weight'], dict_vaccinated_but_not_boosted_matched_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_low_birth_weight['low_birth_weight']/(dict_vaccinated_but_not_boosted_matched_low_birth_weight['normal_birth_weight'] + dict_vaccinated_but_not_boosted_matched_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_boosted_low_birth_weight, dict_vaccinated_but_not_boosted_matched_low_birth_weight)


# evaluate very low birth weight rates
dict_vaccinated_but_not_boosted_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_but_not_boosted)
conversion = dict_vaccinated_but_not_boosted_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_but_not_boosted_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_but_not_boosted_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_but_not_boosted_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_but_not_boosted_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_boosted_very_low_birth_weight = make_very_low_birth_weight_dict(df_boosted)
conversion = dict_boosted_very_low_birth_weight['normal_birth_weight']/sum(dict_boosted_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_boosted_very_low_birth_weight['very_low_birth_weight'], dict_boosted_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_very_low_birth_weight['very_low_birth_weight']/(dict_boosted_very_low_birth_weight['normal_birth_weight'] + dict_boosted_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_boosted_very_low_birth_weight, dict_vaccinated_but_not_boosted_very_low_birth_weight)

dict_vaccinated_but_not_boosted_matched_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_but_not_boosted_matched)
conversion = dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_but_not_boosted_matched_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_but_not_boosted_matched_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_boosted_very_low_birth_weight, dict_vaccinated_but_not_boosted_matched_very_low_birth_weight)


# plot histogram on the number of days between the 2nd and 3rd covid shots
df_boosted = df_boosted
days_between_shots_2_and_3 = list((df_boosted['last_immunization_date'] - df_boosted['full_vaccination_date'])/np.timedelta64(1, 'D') - 14)
plot_histogram(days_between_shots_2_and_3)

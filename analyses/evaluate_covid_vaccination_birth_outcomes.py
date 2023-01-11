# Author: Samantha Piekos
# Date: 11/8/22


# load environment
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import statistics

from datetime import date
from numpy.random import seed
from matplotlib import pyplot as plt
from pyspark.sql.functions import col, lit, unix_timestamp
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.proportion import proportion_confint

!pip install rpy2
import rpy2

%matplotlib inline
plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

seed(31415)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.calculate_fetal_growth_percentile.py
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
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


def count_trimester_of_infection(l):
  l = list(l)
  d = {'1st trimester': 0, '2nd trimester': 0, '3rd trimester': 0}
  for i in l:
    d[i] += 1
  print(d)
  return d


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


def convert_dict_to_df(d, y_label):
  df = pd.DataFrame(columns = ['Group', y_label])
  for k, v in d.items():
    for days in v:
      df = df.append({'Group': k, y_label: days}, ignore_index = True)
  df[y_label] = df[y_label].astype(int)
  return df


def create_violin_plot(d, y_label, list_colors):
  df = convert_dict_to_df(d, y_label)
  ax = sns.violinplot(x='Group', y=y_label, data=df, palette=list_colors)
  ax.set(xlabel=None)
  ax.set_ylim([140, 400])
  plt.show()


# identify the first delivery to a women that was fully vaccinated at time of delivery
temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated")
print(temp.agg({'ob_delivery_delivery_date': 'min'}).show())


# load data
df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_expanded_7").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom")
df_unvaccinated = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date").toPandas().sort_index()
print('# Number of unvaccinated pregnant women who have delivered after 1-25-21: ' + str(len(df_unvaccinated)))

df_vaccinated_mrna_pyspark = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna_expanded_7").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
print('# Number of vaccinated (mRNA) pregnant women who have delivered: ' + str(df_vaccinated_mrna_pyspark.count()))

df_vaccinated_jj = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_jj").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id']).toPandas().sort_index()
print('# Number of vaccinated (J&J) pregnant women who have delivered: ' + str(len(df_vaccinated_jj)))

df_vaccinated_moderna = df_vaccinated_mrna_pyspark.filter(col("first_immunization_name").like("%MODERNA%")).toPandas().sort_index()
print('# Number of women with breakthrough covid infection (Moderna) during pregnancy who has delivered: ' + str(len(df_vaccinated_moderna)))

df_vaccinated_pfizer= df_vaccinated_mrna_pyspark.filter(col("first_immunization_name").like("%PFIZER%")).toPandas().sort_index()
print('# Number of women with breakthrough covid infection (Pfizer) during pregnancy who has delivered: ' + str(len(df_vaccinated_pfizer)))

df_vaccinated_mrna = df_vaccinated_mrna_pyspark.toPandas().sort_index()



# evaluate preterm birth rates and preterm birth classifications
dict_unvaccinated_preterm_status = determine_preterm_status(df_unvaccinated)
dict_unvaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_unvaccinated)
conversion = dict_unvaccinated_term_or_preterm_status['term']/sum(dict_unvaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_term_or_preterm_status['preterm'], dict_unvaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_term_or_preterm_status['preterm']/(dict_unvaccinated_term_or_preterm_status['term'] + dict_unvaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_mrna_vaccinated_preterm_status = determine_preterm_status(df_vaccinated_mrna)
dict_mrna_vaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_mrna)
conversion = dict_mrna_vaccinated_term_or_preterm_status['term']/sum(dict_mrna_vaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_mrna_vaccinated_term_or_preterm_status['preterm'], dict_mrna_vaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_mrna_vaccinated_term_or_preterm_status['preterm']/(dict_mrna_vaccinated_term_or_preterm_status['term'] + dict_mrna_vaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_mrna_vaccinated_preterm_status, dict_unvaccinated_preterm_status))
dict_mrna_vaccinated_preterm_status.pop('term')
dict_unvaccinated_preterm_status.pop('term')
print(calc_fishers_exact_test(dict_mrna_vaccinated_preterm_status, dict_unvaccinated_preterm_status))
print(calc_fishers_exact_test_not_simulated(dict_mrna_vaccinated_term_or_preterm_status, dict_unvaccinated_term_or_preterm_status))

make_bar_chart([dict_mrna_vaccinated_preterm_status, dict_unvaccinated_preterm_status], {'Vaccinated': 'seagreen', 'Unvaccinated': 'crimson'}, standardize=True)

dict_unvaccinated_preterm_status = determine_preterm_status(df_unvaccinated)
dict_unvaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_unvaccinated)
conversion = dict_unvaccinated_term_or_preterm_status['term']/sum(dict_unvaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_term_or_preterm_status['preterm'], dict_unvaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_term_or_preterm_status['preterm']/(dict_unvaccinated_term_or_preterm_status['term'] + dict_unvaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_jj_vaccinated_preterm_status = determine_preterm_status(df_vaccinated_jj)
dict_jj_vaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_jj)
conversion = dict_jj_vaccinated_term_or_preterm_status['term']/sum(dict_jj_vaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_jj_vaccinated_term_or_preterm_status['preterm'], dict_jj_vaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_jj_vaccinated_term_or_preterm_status['preterm']/(dict_jj_vaccinated_term_or_preterm_status['term'] + dict_jj_vaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_jj_vaccinated_preterm_status, dict_unvaccinated_preterm_status))
print(calc_fishers_exact_test_not_simulated(dict_jj_vaccinated_term_or_preterm_status, dict_unvaccinated_term_or_preterm_status))

dict_moderna_vaccinated_preterm_status = determine_preterm_status(df_vaccinated_moderna)
dict_moderna_vaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_moderna)
conversion = dict_moderna_vaccinated_term_or_preterm_status['term']/sum(dict_moderna_vaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_moderna_vaccinated_term_or_preterm_status['preterm'], dict_moderna_vaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_moderna_vaccinated_term_or_preterm_status['preterm']/(dict_moderna_vaccinated_term_or_preterm_status['term'] + dict_moderna_vaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_moderna_vaccinated_preterm_status, dict_unvaccinated_preterm_status))
print(calc_fishers_exact_test_not_simulated(dict_moderna_vaccinated_term_or_preterm_status, dict_unvaccinated_term_or_preterm_status))

dict_pfizer_vaccinated_preterm_status = determine_preterm_status(df_vaccinated_pfizer)
dict_pfizer_vaccinated_term_or_preterm_status = determine_term_or_preterm_status(df_vaccinated_pfizer)
conversion = dict_pfizer_vaccinated_term_or_preterm_status['term']/sum(dict_pfizer_vaccinated_term_or_preterm_status.values())
ci_low, ci_up = proportion_confint(dict_pfizer_vaccinated_term_or_preterm_status['preterm'], dict_pfizer_vaccinated_term_or_preterm_status['term'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_pfizer_vaccinated_term_or_preterm_status['preterm']/(dict_pfizer_vaccinated_term_or_preterm_status['term'] + dict_pfizer_vaccinated_term_or_preterm_status['preterm']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_pfizer_vaccinated_preterm_status, dict_unvaccinated_preterm_status))
print(calc_fishers_exact_test_not_simulated(dict_pfizer_vaccinated_term_or_preterm_status, dict_unvaccinated_term_or_preterm_status))

print(calc_fishers_exact_test(dict_moderna_vaccinated_preterm_status, dict_pfizer_vaccinated_preterm_status))
print(calc_fishers_exact_test_not_simulated(dict_moderna_vaccinated_term_or_preterm_status, dict_pfizer_vaccinated_term_or_preterm_status))


# evaluate gestational days at delivery
temp = [i for i in df_unvaccinated.gestational_days if str(i) != 'nan']
print('Median Unvaccinated Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Unvaccinated Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

temp = [i for i in df_vaccinated_mrna.gestational_days if str(i) != 'nan']
print('Median mRNA Vaccinated Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR mRNA Vaccinated Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_vaccinated_mrna, df_unvaccinated, 'gestational_days'))

dict_days_protected = {'Vaccinated': list(df_vaccinated_mrna.gestational_days), 'Unvaccinated': list(df_unvaccinated.gestational_days)}
list_colors = ['seagreen', 'crimson']
create_violin_plot(dict_days_protected, 'Gestational Days at Delivery', list_colors)

temp = [i for i in df_vaccinated_jj.gestational_days if str(i) != 'nan']
print('Median J&J Vaccinated Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR J&J Vaccinated Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_vaccinated_jj, df_unvaccinated, 'gestational_days'))

temp = [i for i in df_vaccinated_moderna.gestational_days if str(i) != 'nan']
print('Median Moderna Vaccinated Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Moderna Vaccinated Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_vaccinated_moderna, df_unvaccinated, 'gestational_days'))

temp = [i for i in df_vaccinated_pfizer.gestational_days if str(i) != 'nan']
print('Median Pfizer Vaccinated Patients Baby Gestational Days: ' + str(statistics.median(temp)))
print('IQR Pfizer Vaccinated Patients Baby Gestational Days: ' + str(stats.iqr(temp)))
print('\n')

print('Mann Whitney U Test - Gestational Days:')
print(run_mann_whitney_u_test(df_vaccinated_pfizer, df_unvaccinated, 'gestational_days'))

print(run_mann_whitney_u_test(df_vaccinated_moderna, df_vaccinated_pfizer, 'gestational_days'))


# evaluate birth weight
print('Median Unvaccinated Patients Baby Birth Weitht: ' + str(df_unvaccinated.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Unvaccinated Patients Baby Birth Weitht: ' + str(stats.iqr(df_unvaccinated.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print('Median Vaccinated (mRNA) Patients Baby Birth Weitht: ' + str(df_vaccinated_mrna.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated (mRAN) Patients Baby Birth Weitht: ' + str(stats.iqr(df_vaccinated_mrna.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_vaccinated_mrna, df_unvaccinated, 'delivery_infant_birth_weight_oz'))

print('Median Vaccinated (J&J) Patients Baby Birth Weitht: ' + str(df_vaccinated_jj.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated (J&J) Patients Baby Birth Weitht: ' + str(stats.iqr(df_vaccinated_jj.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_vaccinated_jj, df_unvaccinated, 'delivery_infant_birth_weight_oz'))

print('Median Vaccinated (Moderna) Patients Baby Birth Weitht: ' + str(df_vaccinated_moderna.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated (Moderna) Patients Baby Birth Weitht: ' + str(stats.iqr(df_vaccinated_moderna.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_vaccinated_moderna, df_unvaccinated, 'delivery_infant_birth_weight_oz'))

print('Median Vaccinated (Pfizer) Patients Baby Birth Weitht: ' + str(df_vaccinated_pfizer.delivery_infant_birth_weight_oz.dropna().median()))
print('IQR Vaccinated (Pfizer) Patients Baby Birth Weitht: ' + str(stats.iqr(df_vaccinated_pfizer.delivery_infant_birth_weight_oz.dropna().astype(float))))
print('\n')

print(run_mann_whitney_u_test(df_vaccinated_pfizer, df_unvaccinated, 'delivery_infant_birth_weight_oz'))


# evaluate fetal growth percentiles and small for gestational age rates
unvaccinated_fetal_growth_percentiles = get_fetal_growth_percentiles(df_unvaccinated)
print('Median Unvaccinated Patients Baby Fetal Growth Percentile: ' + str(median(unvaccinated_fetal_growth_percentiles)))
print('IQR Unvaccinated Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(unvaccinated_fetal_growth_percentiles)))
print('\n')

dict_unvaccinated_sga = count_small_for_gestational_age(unvaccinated_fetal_growth_percentiles)
conversion = dict_unvaccinated_sga['normal']/sum(dict_unvaccinated_sga.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_sga['SGA'],dict_unvaccinated_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_sga['SGA']/(dict_unvaccinated_sga['normal'] + dict_unvaccinated_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

vaccinated_mrna_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_mrna)
print('Median Vaccinated (mRNA) Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_mrna_fetal_growth_percentiles)))
print('IQR Vaccinated (mRNA) Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_mrna_fetal_growth_percentiles)))
print('\n')
print(stats.mannwhitneyu(vaccinated_mrna_fetal_growth_percentiles, unvaccinated_fetal_growth_percentiles))
print('\n')

dict_vaccinated_mrna_sga = count_small_for_gestational_age(vaccinated_mrna_fetal_growth_percentiles)
conversion = dict_vaccinated_mrna_sga['normal']/sum(dict_vaccinated_mrna_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_mrna_sga['SGA'],dict_vaccinated_mrna_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_mrna_sga['SGA']/(dict_vaccinated_mrna_sga['normal'] + dict_vaccinated_mrna_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_sga, dict_unvaccinated_sga))

vaccinated_jj_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_jj)
print('Median Vaccinated (J&J) Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_jj_fetal_growth_percentiles)))
print('IQR Vaccinated (J&J) Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_jj_fetal_growth_percentiles)))
print('\n')
print(stats.mannwhitneyu(vaccinated_jj_fetal_growth_percentiles, unvaccinated_fetal_growth_percentiles))
print('\n')

dict_vaccinated_jj_sga = count_small_for_gestational_age(vaccinated_jj_fetal_growth_percentiles)
conversion = dict_vaccinated_jj_sga['normal']/sum(dict_vaccinated_jj_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_sga['SGA'],dict_vaccinated_jj_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_sga['SGA']/(dict_vaccinated_jj_sga['normal'] + dict_vaccinated_jj_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_sga, dict_unvaccinated_sga))

vaccinated_moderna_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_moderna)
print('Median Vaccinated (Moderna) Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_moderna_fetal_growth_percentiles)))
print('IQR Vaccinated (Moderna) Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_moderna_fetal_growth_percentiles)))
print('\n')
print(stats.mannwhitneyu(vaccinated_moderna_fetal_growth_percentiles, unvaccinated_fetal_growth_percentiles))
print('\n')

dict_vaccinated_moderna_sga = count_small_for_gestational_age(vaccinated_moderna_fetal_growth_percentiles)
conversion = dict_vaccinated_moderna_sga['normal']/sum(dict_vaccinated_moderna_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_sga['SGA'],dict_vaccinated_moderna_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_sga['SGA']/(dict_vaccinated_moderna_sga['normal'] + dict_vaccinated_moderna_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_sga, dict_unvaccinated_sga))

vaccinated_pfizer_fetal_growth_percentiles = get_fetal_growth_percentiles(df_vaccinated_pfizer)
print('Median Vaccinated (Pfizer) Patients Baby Fetal Growth Percentile: ' + str(median(vaccinated_pfizer_fetal_growth_percentiles)))
print('IQR Vaccinated (Pfizer) Patients Baby Fetal Growth Percentile: ' + str(stats.iqr(vaccinated_pfizer_fetal_growth_percentiles)))
print('\n')
print(stats.mannwhitneyu(vaccinated_pfizer_fetal_growth_percentiles, unvaccinated_fetal_growth_percentiles))
print('\n')

dict_vaccinated_pfizer_sga = count_small_for_gestational_age(vaccinated_pfizer_fetal_growth_percentiles)
conversion = dict_vaccinated_pfizer_sga['normal']/sum(dict_vaccinated_pfizer_sga.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_sga['SGA'],dict_vaccinated_pfizer_sga['normal'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_sga['SGA']/(dict_vaccinated_pfizer_sga['normal'] + dict_vaccinated_pfizer_sga['SGA']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_sga, dict_unvaccinated_sga))


# evaluate stillbirth rates
dict_unvaccinated_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_unvaccinated.iterrows():
  dict_unvaccinated_fetal_death = add_to_fetal_living_status_dict(dict_unvaccinated_fetal_death, row['ob_history_last_known_living_status'])
print(dict_unvaccinated_fetal_death)
dict_unvaccinated_stillbirth = make_stillbirth_dict(df_unvaccinated)
conversion = dict_unvaccinated_stillbirth['Living']/sum(dict_unvaccinated_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_stillbirth['Fetal Demise'], dict_unvaccinated_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_stillbirth['Fetal Demise']/(dict_unvaccinated_stillbirth['Living'] + dict_unvaccinated_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_mrna_vaccinated_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_mrna.iterrows():
  dict_mrna_vaccinated_fetal_death  = add_to_fetal_living_status_dict(dict_mrna_vaccinated_fetal_death, row['ob_history_last_known_living_status'])
print(dict_mrna_vaccinated_fetal_death)
dict_vaccinated_mrna_stillbirth = make_stillbirth_dict(df_vaccinated_mrna)
conversion = dict_vaccinated_mrna_stillbirth['Living']/sum(dict_vaccinated_mrna_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_mrna_stillbirth['Fetal Demise'], dict_vaccinated_mrna_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_mrna_stillbirth['Fetal Demise']/(dict_vaccinated_mrna_stillbirth['Living'] + dict_vaccinated_mrna_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_mrna_vaccinated_fetal_death, dict_unvaccinated_fetal_death))
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_stillbirth, dict_unvaccinated_stillbirth))

dict_jj_vaccinated_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_jj.iterrows():
  dict_jj_vaccinated_fetal_death  = add_to_fetal_living_status_dict(dict_jj_vaccinated_fetal_death, row['ob_history_last_known_living_status'])
print(dict_jj_vaccinated_fetal_death)
dict_vaccinated_jj_stillbirth = make_stillbirth_dict(df_vaccinated_jj)
conversion = dict_vaccinated_jj_stillbirth['Living']/sum(dict_vaccinated_jj_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_stillbirth['Fetal Demise'], dict_vaccinated_jj_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_stillbirth['Fetal Demise']/(dict_vaccinated_jj_stillbirth['Living'] + dict_vaccinated_jj_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_jj_vaccinated_fetal_death, dict_unvaccinated_fetal_death))
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_stillbirth, dict_unvaccinated_stillbirth))

dict_moderna_vaccinated_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_moderna.iterrows():
  dict_moderna_vaccinated_fetal_death  = add_to_fetal_living_status_dict(dict_moderna_vaccinated_fetal_death, row['ob_history_last_known_living_status'])
print(dict_moderna_vaccinated_fetal_death)
dict_vaccinated_moderna_stillbirth = make_stillbirth_dict(df_vaccinated_moderna)
conversion = dict_vaccinated_moderna_stillbirth['Living']/sum(dict_vaccinated_moderna_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_stillbirth['Fetal Demise'], dict_vaccinated_moderna_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_stillbirth['Fetal Demise']/(dict_vaccinated_moderna_stillbirth['Living'] + dict_vaccinated_moderna_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_moderna_vaccinated_fetal_death, dict_unvaccinated_fetal_death))
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_stillbirth, dict_unvaccinated_stillbirth))

dict_pfizer_vaccinated_fetal_death = {'Living': 0, 'None': 0, 'Fetal Demise': 0}
for index, row in df_vaccinated_pfizer.iterrows():
  dict_pfizervaccinated_fetal_death = add_to_fetal_living_status_dict(dict_pfizer_vaccinated_fetal_death, row['ob_history_last_known_living_status'])
print(dict_pfizer_vaccinated_fetal_death)
dict_vaccinated_pfizer_stillbirth = make_stillbirth_dict(df_vaccinated_pfizer)
conversion = dict_vaccinated_pfizer_stillbirth['Living']/sum(dict_vaccinated_pfizer_stillbirth.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_stillbirth['Fetal Demise'], dict_vaccinated_pfizer_stillbirth['Living'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_stillbirth['Fetal Demise']/(dict_vaccinated_pfizer_stillbirth['Living'] + dict_vaccinated_pfizer_stillbirth['Fetal Demise']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

print(calc_fishers_exact_test(dict_pfizer_vaccinated_fetal_death, dict_unvaccinated_fetal_death))
print(calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_stillbirth, dict_unvaccinated_stillbirth))

print(calc_fishers_exact_test(dict_vaccinated_moderna_stillbirth, dict_vaccinated_pfizer_stillbirth))


# evaluate low birth weight rates
dict_unvaccinated_low_birth_weight = make_low_birth_weight_dict(df_unvaccinated)
conversion = dict_unvaccinated_low_birth_weight['normal_birth_weight']/sum(dict_unvaccinated_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_low_birth_weight['low_birth_weight'], dict_unvaccinated_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_low_birth_weight['low_birth_weight']/(dict_unvaccinated_low_birth_weight['normal_birth_weight'] + dict_unvaccinated_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_vaccinated_mrna_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_mrna)
conversion = dict_vaccinated_mrna_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_mrna_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_mrna_low_birth_weight['low_birth_weight'], dict_vaccinated_mrna_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_mrna_low_birth_weight['low_birth_weight']/(dict_vaccinated_mrna_low_birth_weight['normal_birth_weight'] + dict_vaccinated_mrna_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_low_birth_weight, dict_unvaccinated_low_birth_weight)

dict_vaccinated_jj_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_jj)
conversion = dict_vaccinated_jj_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_jj_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_low_birth_weight['low_birth_weight'], dict_vaccinated_jj_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_low_birth_weight['low_birth_weight']/(dict_vaccinated_jj_low_birth_weight['normal_birth_weight'] + dict_vaccinated_jj_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')
calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_low_birth_weight, dict_unvaccinated_low_birth_weight)

dict_vaccinated_moderna_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_moderna)
conversion = dict_vaccinated_moderna_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_moderna_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_low_birth_weight['low_birth_weight'], dict_vaccinated_moderna_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_low_birth_weight['low_birth_weight']/(dict_vaccinated_moderna_low_birth_weight['normal_birth_weight'] + dict_vaccinated_moderna_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_low_birth_weight, dict_unvaccinated_low_birth_weight)

dict_vaccinated_pfizer_low_birth_weight = make_low_birth_weight_dict(df_vaccinated_pfizer)
conversion = dict_vaccinated_pfizer_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_pfizer_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_low_birth_weight['low_birth_weight'], dict_vaccinated_pfizer_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_low_birth_weight['low_birth_weight']/(dict_vaccinated_pfizer_low_birth_weight['normal_birth_weight'] + dict_vaccinated_pfizer_low_birth_weight['low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_low_birth_weight, dict_unvaccinated_low_birth_weight)


# evaluate very low birth weight rates
dict_unvaccinated_very_low_birth_weight = make_very_low_birth_weight_dict(df_unvaccinated)
conversion = dict_unvaccinated_very_low_birth_weight['normal_birth_weight']/sum(dict_unvaccinated_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_very_low_birth_weight['very_low_birth_weight'], dict_unvaccinated_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_very_low_birth_weight['very_low_birth_weight']/(dict_unvaccinated_very_low_birth_weight['normal_birth_weight'] + dict_unvaccinated_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_vaccinated_mrna_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_mrna)
conversion = dict_vaccinated_mrna_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_mrna_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_mrna_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_mrna_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_mrna_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_mrna_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_mrna_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_mrna_very_low_birth_weight, dict_unvaccinated_very_low_birth_weight)

dict_vaccinated_jj_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_jj)
conversion = dict_vaccinated_jj_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_jj_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_jj_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_jj_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_jj_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_jj_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_jj_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_jj_very_low_birth_weight, dict_unvaccinated_very_low_birth_weight)

dict_vaccinated_moderna_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_moderna)
conversion = dict_vaccinated_moderna_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_moderna_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_moderna_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_moderna_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_moderna_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_moderna_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_moderna_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_very_low_birth_weight, dict_unvaccinated_very_low_birth_weight)

dict_vaccinated_pfizer_very_low_birth_weight = make_very_low_birth_weight_dict(df_vaccinated_pfizer)
conversion = dict_vaccinated_pfizer_very_low_birth_weight['normal_birth_weight']/sum(dict_vaccinated_pfizer_very_low_birth_weight.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_pfizer_very_low_birth_weight['very_low_birth_weight'], dict_vaccinated_pfizer_very_low_birth_weight['normal_birth_weight'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_pfizer_very_low_birth_weight['very_low_birth_weight']/(dict_vaccinated_pfizer_very_low_birth_weight['normal_birth_weight'] + dict_vaccinated_pfizer_very_low_birth_weight['very_low_birth_weight']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

calc_fishers_exact_test_not_simulated(dict_vaccinated_pfizer_very_low_birth_weight, dict_unvaccinated_very_low_birth_weight)

calc_fishers_exact_test_not_simulated(dict_vaccinated_moderna_very_low_birth_weight, dict_vaccinated_pfizer_very_low_birth_weight)

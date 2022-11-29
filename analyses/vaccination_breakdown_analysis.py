# Author: Samantha Piekos
# Date: 11/2/22

# load environment
from datetime import date, datetime, timedelta
from pyspark.sql.functions import unix_timestamp
from dateutil.relativedelta import *
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statistics
from sklearn.impute import SimpleImputer
!pip install lifelines
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.anova import AnovaRM
dbutils.library.installPyPI("rpy2")
import rpy2
plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42


# declare universal variables
DATE_START = date(2021, 1, 26)
DATE_START_COVID = date(2020, 6, 18)
CUTOFF_WILDTYPE_START = datetime.datetime(2020, 3, 5) # when testing first became available in the US
CUTOFF_ALPHA_START = datetime.datetime(2021, 4, 24) # alpha first reported as >50% of covid cases in the US by the CDC https://www.cdc.gov/mmwr/volumes/70/wr/mm7023a3.htm?s_cid=mm7023a3_w
CUTOFF_DELTA_START = datetime.datetime(2021, 7, 3)  # delta first reported as >50% of covid cases in the western US by the CDC
CUTOFF_OMICRON_START = datetime.datetime(2021, 12, 25)  # omicron first reported as >50% of covid cases in the western US by the CDC
CUTOFF_OMICRON_BA2_START = datetime.datetime(2022, 3, 26)  # omicron BA.2 first reported as >50% of covid cases in the western US by the CDC
CUTOFF_OMICRON_BA2_12_1_START = datetime.datetime(2022, 5, 28)  # omicron BA2.12.1 first reported as >50% of covid cases in the western US by the CDC
CUTOFF_OMICRON_BA5_START = datetime.datetime(2022, 7, 2)  # omicron BA5 first reported as >50% of covid cases in the western US by the CDC
#LIST_MONTHS_STUDIED = [datetime(2020, 3, 5), datetime(2020, 4, 5), datetime(2020, 5, 5), datetime(2020, 6, 5), datetime(2020, 7, 5), datetime(2020, 8, 5), datetime(2020, 9, 5), datetime(2020, 10, 5), datetime(2020, 11, 5), datetime(2020, 12, 5), datetime(2021, 1, 5), datetime(2021, 2, 5), datetime(2021, 3, 5), datetime(2021, 4, 5), datetime(2021, 5, 5), datetime(2021, 6, 5), datetime(2021, 7, 5), datetime(2021, 8, 5), datetime(2021, 9, 5), datetime(2021, 10, 5), datetime(2021, 11, 5), datetime(2021, 12, 5), datetime(2022, 1, 5), datetime(2022, 2, 5), datetime(2022, 3, 5)]


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.general
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
def verify_if_date_during_pregnancy(test_date, conception_date, delivery_date):
  test_date = test_date.date()
  if conception_date < test_date < delivery_date:
    return test_date
  return None


def verify_if_date_prior_to_pregnancy(test_date, conception_date):
  test_date = test_date.date()
  if conception_date > test_date:
    return test_date
  return None


def check_positive_test_timing(test_dates, test_results, conception_date, delivery_date):
  i = 0
  covid_test_date = None
  covid_immunity_test_date = None
  for test_result in test_results:
    if test_result == 'positive':
      test_date = test_dates[i]
      covid_during_pregnancy_date = verify_if_date_during_pregnancy(test_date, conception_date, delivery_date)
      covid_prior_to_pregnancy_date = verify_if_date_prior_to_pregnancy(test_date, conception_date)
      if covid_prior_to_pregnancy_date:
        covid_immunity_test_date = covid_prior_to_pregnancy_date
      if covid_during_pregnancy_date:
        covid_test_date = covid_during_pregnancy_date
        return(covid_test_date, covid_immunity_test_date)
    i += 1
  return(covid_test_date, covid_immunity_test_date)


def create_pyspark_dataframe(df_original, df_covid_test):
  df_original.createOrReplaceTempView("original")
  df_covid_test.createOrReplaceTempView("covid_test")
  df_filtered_final = spark.sql(
  """
  FROM original AS o
  INNER JOIN covid_test AS c
  ON o.pat_id == c.pat_id
    AND o.episode_id == c.episode_id
    AND o.child_episode_id == c.child_episode_id
    SELECT o.*, c.covid_test_date
  """).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
  return df_filtered_final


def create_pyspark_dataframe_2(df_original, df_covid_test):
  df_original.createOrReplaceTempView("original")
  df_covid_test.createOrReplaceTempView("covid_test")
  df_filtered_final = spark.sql(
  """
  FROM original AS o
  INNER JOIN covid_test AS c
  ON o.pat_id == c.pat_id
    AND o.episode_id == c.episode_id
    AND o.child_episode_id == c.child_episode_id
    SELECT o.*, c.covid_induced_immunity_date
  """).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
  return df_filtered_final


def get_covid_test_date(df):
  df_filtered = pd.DataFrame(columns=['pat_id', 'episode_id', 'child_episode_id', 'covid_test_date' 'conception_date'])
  df_covid_immunity = pd.DataFrame(columns=['pat_id', 'episode_id', 'child_episode_id', 'covid_induced_immunity_date', 'conception_date'])
  df_pandas = df.toPandas()
  for index, row in df_pandas.iterrows():
    conception_date = row['conception_date']
    delivery_date = row['ob_delivery_delivery_date']
    test_dates = list(row['ordering_datetime_collect_list'])
    test_results = list(row['result_short_collect_list'])
    covid_test_date, covid_immunity_test_date = check_positive_test_timing(test_dates, test_results, conception_date, delivery_date)
    if covid_test_date:
      df_filtered = df_filtered.append({'pat_id': row['pat_id'],\
                                        'episode_id': row['episode_id'],\
                                        'child_episode_id': row['child_episode_id'],\
                                        'covid_test_date': covid_test_date,\
                                        'conception_date': row['conception_date']},\
                                        ignore_index=True)
    if covid_immunity_test_date:
      df_covid_immunity = df_covid_immunity.append({'pat_id': row['pat_id'],\
                                        'episode_id': row['episode_id'],\
                                        'child_episode_id': row['child_episode_id'],\
                                        'covid_induced_immunity_date': covid_immunity_test_date,\
                                        'conception_date': row['conception_date']},\
                                        ignore_index=True)
  df_filtered_final = create_pyspark_dataframe(df, spark.createDataFrame(df_filtered))
  df_covid_immunity_final = create_pyspark_dataframe_2(df, spark.createDataFrame(df_covid_immunity))
  df_covid_immunity_final = df_covid_immunity_final.withColumn('days_from_covid_to_conception', datediff('conception_date', 'covid_induced_immunity_date'))
  return(df_filtered_final, df_covid_immunity_final)


def run_mann_whitney_u_test(df_1, df_2, k):
  data_1, data_2 = [], []
  for index, row in df_1.iterrows():
    if row[k] is not None:
      data_1.append(float(row[k]))
  for index, row in df_2.iterrows():
    if row[k] is not None:
      data_2.append(float(row[k]))
  return(stats.mannwhitneyu(data_1, data_2))


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
  
  
def calc_fishers_exact_test(dict_obs, dict_exp):
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
  res = stats.fisher_test(x=list_contingency, simulate_p_value=True)
  print(res)


def make_n_violin_box_plot(dict_outliers):
  dict_filtered = {}
  for k, v in dict_outliers.items():
      if len(v) > 9:
          dict_filtered[k] = v
  fig, ax = plt.subplots()

  # Create a plot
  ax.violinplot(list(dict_filtered.values()))
  ax.boxplot(list(dict_filtered.values()))

  # add x-tick labels
  xticklabels = dict_filtered.keys()
  ax.set_ylabel('Number of Days\nfrom Full Vaccination Status')
  ax.set_xticks([1,2])
  ax.set_xticklabels(xticklabels, rotation = 15)
  ax.set_ylim([0, 650])
  ax.hlines(182, xmin=0, xmax=3, colors='black')
  plt.show()


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
  ax.set_ylim([0, 700])
  plt.show()


def make_variant_covid_infection_dict(df):
  df = df.toPandas()
  d = {'Wild-type': 0, 'Alpha': 0, 'Delta': 0, 'Omicron': 0, 'Omicron BA.2': 0, 'Omicron BA.2.12.1': 0, 'Omicron BA.5': 0}
  for index, row in df.iterrows():
    covid_test_date = row['covid_test_date']
    if covid_test_date >= CUTOFF_OMICRON_BA5_START.date():
      d['Omicron BA.5'] += 1
    elif covid_test_date >= CUTOFF_OMICRON_BA2_12_1_START.date():
      d['Omicron BA.2.12.1'] += 1
    elif covid_test_date >= CUTOFF_OMICRON_BA2_START.date():
      d['Omicron BA.2'] += 1
    elif covid_test_date >= CUTOFF_OMICRON_START.date():
      d['Omicron'] += 1
    elif covid_test_date >= CUTOFF_DELTA_START.date():
      d['Delta'] += 1
    elif covid_test_date >= CUTOFF_ALPHA_START.date():
      d['Alpha'] += 1
    else:
      d['Wild-type'] += 1
  print(d)
  return d


def calc_delivery_date_days_from_start(list_date_delivery):
  n_days = []
  for date_delivery in list_date_delivery:
    dif = (date_delivery.date() - DATE_START).days
    n_days.append(dif)
  return n_days

    
def format_data_for_delivery_chart(data):
  data_final = []
  for df in data:
    df = df.toPandas()
    n_days = calc_delivery_date_days_from_start(list(df['ob_delivery_delivery_date']))
    data_final.append(n_days)
  return data_final
    

def plot_vaccination_status_at_delivery_chart(list_data, list_colors):
  n_bins = 91
  x_min, x_max = 0, 637
  x_ticks = ("Jan 26, '21", "Feb 23, '21", "Mar 23, '21", "Apr 20, '21", "May 18, '21", "Jun 15, '21", "Jul 13, '21", "Aug 10, '21", "Sep 7, '21", "Oct 5, '21", "Nov 2, '21", "Nov 30, '21", "Dec 28, '21", "Jan 25, '22", "Feb 22, '22", "Mar 22, '22", "Apr 19, '22", "May 17, '22", "Jun 14, '22", "Jul 12, '22", "Aug 9, '22", "Sep 6, '22", "Oct 4, '22") 
  c=0
  c=0
  
  for data in list_data:  # plot different vaccination groups
    color = list_colors[c]
    y, binEdges = np.histogram(data, bins=n_bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c=color)
    c+=1
  
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min, x_max, 28.0), x_ticks, fontsize=12, rotation=75)
  plt.ylabel("Number of Patients Delivering")
  plt.tight_layout()
  plt.show()


def initiate_dict(x_max):
  d = {}
  c = 0
  while c < x_max:
    d[c] = 0
    c += 1
  return d


def make_delivery_date_days_from_start_dict(list_date_delivery, x_max):
  d = initiate_dict(x_max)
  for date_delivery in list_date_delivery:
    dif = (date_delivery.date() - DATE_START).days
    if dif < x_max:
      d[dif] += 1
  return d


def calc_proportion_with_covid_at_delivery(dict_delivery, dict_covid, x_max):
  list_proportion = []
  n_delivery = 0
  n_covid = 0
  c = 0
  while c < x_max:
    n_delivery += dict_delivery[c]
    n_covid += dict_covid[c]
    if c%7 == 0:
      if n_delivery > 9:
        list_proportion.append(100*n_covid/n_delivery)
      else:
        list_proportion.append(0)
      n_delivery = 0
      n_covid = 0
    c += 1
  return list_proportion

    
def format_data_for_chart(data):
  data_final = []
  x_max = 637
  for df_delivery, df_covid in data:
    df_delivery = df_delivery.toPandas()
    dict_delivery = make_delivery_date_days_from_start_dict(list(df_delivery['ob_delivery_delivery_date']), x_max)
    df_covid = df_covid.toPandas()
    dict_covid = make_delivery_date_days_from_start_dict(list(df_covid['ob_delivery_delivery_date']), x_max)
    list_proportion = calc_proportion_with_covid_at_delivery(dict_delivery, dict_covid, x_max)
    data_final.append(list_proportion)
  return data_final


def plot_covid_infection_chart(list_data, list_colors):
  y_min, y_max = 0, 100
  x_min, x_max = 0, 637
  x_ticks = ("Jan 26, '21", "Feb 23, '21", "Mar 23, '21", "Apr 20, '21", "May 18, '21", "Jun 15, '21", "Jul 13, '21", "Aug 10, '21", "Sep 7, '21", "Oct 5, '21", "Nov 2, '21", "Nov 30, '21", "Dec 28, '21", "Jan 25, '22", "Feb 22, '22", "Mar 22, '22", "Apr 19, '22", "May 17, '22", "Jun 14, '22", "Jul 12, '22", "Aug 9, '22", "Sep 6, '22", "Oct 4, '22") 
  c=0
  
  for data in list_data:  # plot different vaccination groups
    color = list_colors[c]
    plt.plot(np.arange(x_min, x_max, 7), data, '-', c=color)
    c+=1
  
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min, x_max, 28.0), x_ticks, fontsize=12, rotation=75)
  plt.ylim(y_min, y_max)
  plt.ylabel("Percentage of People at Delivery \nwith COVID-19 during Pregnancy")
  plt.tight_layout()
  plt.show()

  
def add_list_to_df(df, l, cond):
  c = 0
  for item in l:
    d = {'sub_id': c, 'y': item, 'cond': cond}
    df = df.append(d, ignore_index=True)
    c += 1
  return df
  
  
def perform_anova(l_experimental, l_control):
  df = pd.DataFrame(columns=['sub_id', 'y', 'cond'])
  df = add_list_to_df(df, l_experimental, 'experimental')
  df = add_list_to_df(df, l_control, 'control')
  aovrm = AnovaRM(df, 'y', 'sub_id', within=['cond'])
  res = aovrm.fit()
  return res


def calc_gestational_day_of_full_vaccination_status(df):
  l_days = []
  for index, row in df.iterrows():
    l_days.append(row['gestational_days'] - (row['ob_delivery_delivery_date'].date() - row['full_vaccination_date'].date()).days)
  return l_days


def calc_gestational_day_of_booster_shot(df):
  l_days = []
  for index, row in df.iterrows():
    l_days.append(row['gestational_days'] - (row['ob_delivery_delivery_date'].date() - row['booster_date'].date()).days)
  return l_days


def determine_trimester(days):
  if days <= 84:
    return "1st trimester"
  elif days <= 182:
    return "2nd trimester"
  return "3rd trimester"


def determine_percent_of_pop(d, size):
  print('# Percentage of Patients Acheiving Full Vaccination Status in the 1st Trimester of Preganncy: ' + str(100*d['1st trimester']/size) + '%')
  print('# Percentage of Patients Acheiving Full Vaccination Status in the 2nd Trimester of Preganncy: ' + str(100*d['2nd trimester']/size)+ '%')
  print('# Percentage of Patients Acheiving Full Vaccination Status in the 3rd Trimester of Preganncy: ' + str(100*d['3rd trimester']/size)+ '%')


def calc_trimester_of_full_vaccination_status(l):
  d = {'1st trimester': 0, '2nd trimester': 0, '3rd trimester': 0}
  size = len(l)
  for i in l:
    trimester = determine_trimester(i)
    d[trimester] += 1
  print(d)
  determine_percent_of_pop(d, size)
  return d


def plot_histogram(data, color, name):
  n_bins = 42
  x_min, x_max = 0, 294
  x_ticks = ['4', '8', '12', '16', '20', '24', '28', '32', '36', '40']
  
  plt.hist(data, bins=n_bins, range=(x_min, x_max), color=color)
  plt.axvline(x=84, color='k', linestyle='--')
  plt.axvline(x=182, color='k', linestyle='--')
  #plt.vlines(84, ymin=0, ymax=550, color='black')
  #plt.vlines(182, ymin=0, ymax=550, color='black')
  
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min+28, x_max, 28.0), x_ticks, fontsize=12, rotation=0)
  plt.xlabel("Gestational Week During Which\n"+name)
  plt.ylabel("Number of Pregnant People")
  plt.tight_layout()
  plt.show()


# load immunization table
df_immunization = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_immunization")
df_immunization.createOrReplaceTempView("immunization")


# define cohorts
# filter maternity cohort for pregnant people that delivered >140 gestation, between the ages of 18-45, singleton pregnancy, and no recorded covid-19 infection recorded prior to pregnancy
df_cohort_maternity_covid_immunity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_immunity")
df_cohort_maternity_covid_immunity.createOrReplaceTempView("covid_immunity_mom")
df_mom = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity AS m WHERE m.number_of_fetuses == 1 and m.age_at_start_dt >= 18 and m.age_at_start_dt < 45 and m.gestational_days >= 140 and m.ob_delivery_delivery_date >= '2020-03-05'").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_mom = df_mom.withColumn('conception_date', expr("date_sub(ob_delivery_delivery_date, gestational_days)"))
df_mom.createOrReplaceTempView("mom")
df_mom = spark.sql(
"""
FROM mom AS mom
LEFT ANTI JOIN covid_immunity_mom AS cim
ON cim.pat_id = mom.pat_id
  AND cim.instance = mom.instance
  AND cim.episode_id = mom.episode_id
  AND cim.child_episode_id = mom.child_episode_id
SELECT mom.*
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

df_mom.createOrReplaceTempView("mom")


# Identify pregnant women fully vaccinated with Moderna
df_cohort_mom_moderna = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 2) + interval '14' day < m.ob_delivery_delivery_date
  AND element_at(array_distinct(i.all_immunization_dates), 1) != element_at(array_distinct(i.all_immunization_dates), 2)
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, element_at(array_distinct(i.all_immunization_dates), 2) + interval '14' day AS full_vaccination_date, element_at(i.all_immunization_names, 2) AS full_vaccination_name, i.all_immunization_dates, i.all_immunization_names
""").filter(col("first_immunization_name").like("%MODERNA%")).filter(col("full_vaccination_name").like("%MODERNA%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# identify pregnant people that were boosted (moderna) prior to delivery
df_cohort_mom_moderna.createOrReplaceTempView("moderna")
df_cohort_mom_moderna_boosted_1 = spark.sql("SELECT moderna.*, element_at(array_distinct(moderna.all_immunization_dates), 3) AS booster_date, element_at(array_distinct(moderna.all_immunization_names), 3) AS booster_name FROM moderna WHERE (full_vaccination_date - interval '14' day) != element_at(array_distinct(moderna.all_immunization_dates), 3) AND element_at(array_distinct(moderna.all_immunization_dates), 3) < ob_delivery_delivery_date")
df_cohort_mom_moderna_boosted_2 = spark.sql("SELECT moderna.*, last_immunization_date AS booster_date, last_immunization_name AS booster_name  FROM moderna WHERE (full_vaccination_date - interval '14' day) != last_immunization_date AND last_immunization_date < ob_delivery_delivery_date")
df_cohort_mom_moderna_boosted = df_cohort_mom_moderna_boosted_1.union(df_cohort_mom_moderna_boosted_2).filter(~col("booster_name").like("%BIVALENT%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

# identify pregnant people that only received the full vaccination moderna series prior to delivery
df_cohort_mom_moderna_boosted.createOrReplaceTempView("moderna_boosted")
df_cohort_mom_moderna_2_shots_only = spark.sql(
"""
FROM moderna AS m
LEFT ANTI JOIN moderna_boosted AS mb
ON m.pat_id == mb.pat_id
  AND m.episode_id == mb.episode_id
  AND m.child_episode_id == mb.child_episode_id
SELECT m.*
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# Identify pregnant women fully vaccinated with Pfizer
df_cohort_mom_pfizer = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 1) != element_at(array_distinct(i.all_immunization_dates), 2)
  AND element_at(array_distinct(i.all_immunization_dates), 2) + interval '14' day < m.ob_delivery_delivery_date
  AND i.last_immunization_date > i.first_immunization_date
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, element_at(array_distinct(i.all_immunization_dates), 2) + interval '14' day AS full_vaccination_date, element_at(i.all_immunization_names, 2) AS full_vaccination_name, i.all_immunization_dates, i.all_immunization_names
""").filter(col("first_immunization_name").like("%PFIZER%")).filter(col("full_vaccination_name").like("%PFIZER%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

# identify pregnant people that were boosted (pfizer) prior to delivery
df_cohort_mom_pfizer.createOrReplaceTempView("pfizer")
df_cohort_mom_pfizer_boosted_1 = spark.sql("SELECT pfizer.*, element_at(array_distinct(pfizer.all_immunization_dates), 3) AS booster_date, element_at(array_distinct(pfizer.all_immunization_names), 3) AS booster_name FROM pfizer WHERE (full_vaccination_date - interval '14' day) != element_at(array_distinct(pfizer.all_immunization_dates), 3) AND element_at(array_distinct(pfizer.all_immunization_dates), 3) < ob_delivery_delivery_date")
df_cohort_mom_pfizer_boosted_2 = spark.sql("SELECT pfizer.*, last_immunization_date AS booster_date, last_immunization_name AS booster_name FROM pfizer WHERE (full_vaccination_date - interval '14' day) != last_immunization_date AND last_immunization_date < ob_delivery_delivery_date")
df_cohort_mom_pfizer_boosted = df_cohort_mom_pfizer_boosted_1.union(df_cohort_mom_pfizer_boosted_2).filter(~col("booster_name").like("%BIVALENT%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

# identify pregnant people that only received the full vaccination moderna series prior to delivery
df_cohort_mom_pfizer_boosted.createOrReplaceTempView("pfizer_boosted")
df_cohort_mom_pfizer_2_shots_only = spark.sql(
"""
FROM pfizer AS p
LEFT ANTI JOIN pfizer_boosted AS pb
ON p.pat_id == pb.pat_id
  AND p.episode_id == pb.episode_id
  AND p.child_episode_id == pb.child_episode_id
SELECT p.*
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# create single cohrot of fully vaccinated with or without mRNA booster shots at delivery
df_cohort_mom_mrna_2_shots_only = df_cohort_mom_pfizer_2_shots_only.union(df_cohort_mom_moderna_2_shots_only).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
write_data_frame_to_sandbox(df_cohort_mom_mrna_2_shots_only, 'snp2_cohort_maternity_mrna_2_shots_only', sandbox_db='rdp_phi_sandbox', replace=True)
df_cohort_mom_mrna_boosted = df_cohort_mom_pfizer_boosted.union(df_cohort_mom_moderna_boosted).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
write_data_frame_to_sandbox(df_cohort_mom_mrna_boosted, 'snp2_cohort_maternity_mrna_boosted', sandbox_db='rdp_phi_sandbox', replace=True)

# create vaccinated, but not boosted cohort
df_cohort_mom_mrna_2_shots_only.createOrReplaceTempView("two_mrna_shots_only")
df_vaccinated_but_not_boosted = spark.sql("SELECT tmso.* FROM two_mrna_shots_only AS tmso WHERE (tmso.full_vaccination_date + interval '168' day) < tmso.ob_delivery_delivery_date AND tmso.ob_delivery_delivery_date > '2021-09-22'")
write_data_frame_to_sandbox(df_vaccinated_but_not_boosted, 'snp2_cohort_maternity_vaccinated_but_not_boosted', sandbox_db='rdp_phi_sandbox', replace=True)

# create initial series + bivalent booster and inital series + booster + bivalent booster cohorts
df_cohort_mom_moderna_boosted = df_cohort_mom_moderna_boosted_1.union(df_cohort_mom_moderna_boosted_2).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_pfizer_boosted = df_cohort_mom_pfizer_boosted_1.union(df_cohort_mom_pfizer_boosted_2).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_mrna_boosted = df_cohort_mom_pfizer_boosted.union(df_cohort_mom_moderna_boosted).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_vaccinated_bivalent_boosted = df_cohort_mom_mrna_boosted.filter(col("booster_name").like("%BIVALENT%"))
write_data_frame_to_sandbox(df_cohort_mom_vaccinated_bivalent_boosted, 'snp2_cohort_maternity_vaccinated_bivalent_boosted', sandbox_db='rdp_phi_sandbox', replace=True)
df_cohort_mom_vaccinated_boosted_bivalent_boosted = df_cohort_mom_mrna_boosted.filter(~col("booster_name").like("%BIVALENT%")).filter(col("last_immunization_name").like("%BIVALENT%"))
write_data_frame_to_sandbox(df_cohort_mom_vaccinated_boosted_bivalent_boosted, 'snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted', sandbox_db='rdp_phi_sandbox', replace=True)


# identify pregnant people with only 1 mrna shot at delivery

# find pregnant people that only received 1 shot of moderna
df_cohort_mom_moderna_1_shot = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 1) < m.ob_delivery_delivery_date
  AND size(array_distinct(i.all_immunization_dates)) == 1
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, i.all_immunization_dates
""").filter(col("first_immunization_name").like("%MODERNA%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# find pregnant people that received more than 1 shot of moderna, but only 1 dose was administered prior to delivery
df_cohort_mom_moderna_1_shot_2 = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 1) < m.ob_delivery_delivery_date
  AND element_at(array_distinct(i.all_immunization_dates), 2) > m.ob_delivery_delivery_date
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, i.all_immunization_dates
""").filter(col("first_immunization_name").like("%MODERNA%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# find pregnant people that only received 1 shot of pfizer
df_cohort_mom_pfizer_1_shot = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 1) < m.ob_delivery_delivery_date
  AND size(array_distinct(i.all_immunization_dates)) == 1
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, i.all_immunization_dates
""").filter(col("first_immunization_name").like("%PFIZER%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# find pregnant people that received more than 1 shot of pfizer, but only 1 dose was administered prior to delivery
df_cohort_mom_pfizer_1_shot_2 = spark.sql(
"""
FROM immunization AS i
INNER JOIN mom AS m
ON i.pat_id = m.pat_id
  AND element_at(array_distinct(i.all_immunization_dates), 1) < m.ob_delivery_delivery_date
  AND element_at(array_distinct(i.all_immunization_dates), 2) > m.ob_delivery_delivery_date
SELECT m.*, i.first_immunization_date, i.first_immunization_name, i.last_immunization_date, i.last_immunization_name, i.all_immunization_dates
""").filter(col("first_immunization_name").like("%PFIZER%")).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id']).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])


# combine all to form cohort of pregnant people that only received 1 shot prior to delivery
df_cohort_mom_mrna_1_shot_only = df_cohort_mom_moderna_1_shot.union(df_cohort_mom_moderna_1_shot_2).union(df_cohort_mom_pfizer_1_shot).union(df_cohort_mom_pfizer_1_shot_2).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
write_data_frame_to_sandbox(df_cohort_mom_mrna_1_shot_only, 'snp2_cohort_maternity_mrna_1_shot_only', sandbox_db='rdp_phi_sandbox', replace=True)


df_cohort_mom_mrna_vaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna")
df_cohort_mom_mrna_vaccinated.createOrReplaceTempView("mrna_vaccinated")
print('# Number of people fully vaccinated (mRNA) at delivery regardless of boosted status: ' + str(df_cohort_mom_mrna_vaccinated.count()))

df_cohort_mom_mrna_2_shots = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only")
df_cohort_mom_mrna_2_shots.createOrReplaceTempView("mrna_2_shots")
print('# Number of people with only 2 shots (mrna) at delivery: ' + str(df_cohort_mom_mrna_2_shots.count()))

df_cohort_mom_mrna_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted")
df_cohort_mom_mrna_boosted.createOrReplaceTempView("mrna_boosted")
print('# Number of people boosted (mrna) at delivery: ' + str(df_cohort_mom_mrna_boosted.count()))
print('\n')


df_cohort_mom_moderna_2_shots = df_cohort_mom_mrna_2_shots.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of people with only 2 shots (moderna) at delivery: ' + str(df_cohort_mom_moderna_2_shots.count()))

df_cohort_mom_moderna_boosted = df_cohort_mom_mrna_boosted.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of people boosted (moderna) at delivery: ' + str(df_cohort_mom_moderna_boosted.count()))
print('\n')


df_cohort_mom_pfizer_2_shots = df_cohort_mom_mrna_2_shots.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of people with only 2 shots (pfizer) at delivery: ' + str(df_cohort_mom_pfizer_2_shots.count()))

df_cohort_mom_pfizer_boosted = df_cohort_mom_mrna_boosted.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of people boosted (pfizer) at delivery: ' + str(df_cohort_mom_pfizer_boosted.count()))
print('\n')


df_cohort_mom_vaccinated_but_not_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted")
df_cohort_mom_vaccinated_but_not_boosted.createOrReplaceTempView("vaccinated_but_not_boosted")
print('# Number of pregnant people that were vaccinated, but not boosted where initial series ended >6 months prior: ' + str(df_cohort_mom_vaccinated_but_not_boosted.count()))

df_cohort_mom_vaccianted_bivalent_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_bivalent_boosted")
df_cohort_mom_vaccianted_bivalent_boosted.createOrReplaceTempView("vaccinated_bivalent_boosted")
print('# Number of pregnant people that were vaccinated and bivalent boosted, but not boosted: ' + str(df_cohort_mom_vaccianted_bivalent_boosted.count()))

df_cohort_mom_vaccianted_boosted_bivalent_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted")
df_cohort_mom_vaccianted_boosted_bivalent_boosted.createOrReplaceTempView("vaccinated_boosted_bivalent_boosted")
print('# Number of pregnant people that were vaccinated, boosted, and bivalent boosted: ' + str(df_cohort_mom_vaccianted_boosted_bivalent_boosted.count()))
print('\n')


df_cohort_mom_mrna_1_shot = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_only")
df_cohort_mom_mrna_1_shot.createOrReplaceTempView("mrna_1_shot")
print('# Number of people vaccinated with only 1 mRNA shot prior to delivery: ' + str(df_cohort_mom_mrna_1_shot.count()))
print('\n')


df_cohort_mom_jj = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_jj")
df_cohort_mom_jj.createOrReplaceTempView("jj")
print('# Number of people vaccinated with J&J prior to delivery: ' + str(df_cohort_mom_jj.count()))
print('\n')


df_covid_induced_immunity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_unvaccinated_covid_immunity")
df_covid_induced_immunity.createOrReplaceTempView("covid_induced_immunity")
print('# Number of unvaccinated pregnant people that had covid prior to their pregnancy: ' + str(df_covid_induced_immunity.count()))


df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated")
print('# Number of unvaccinated pregnant women who have delivered: ' + str(df_unvaccinated.count()))
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom")
df_unvaccinated_limited = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")
print('# Number of unvaccinated pregnant people who have delivered after 1-25-21: ' + str(df_unvaccinated_limited.count()))
print('\n')


df_cohort_covid_maternity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_expanded_6").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_covid_maternity.createOrReplaceTempView("covid_mom")
print('# Number of pregnant people that have had COVID-19 at somepoint regardless of vaccination status: ' + str(df_cohort_covid_maternity.count()))

df_covid_mom = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_test_data")
df_covid_mom.createOrReplaceTempView("covid_mom")
print('# Number of pregnant people with COVID-19 during pregnancy regardless of vaccination status: ' + str(df_covid_mom.count()))
print('\n')


# identify those that had a breakthrough infection that had only gotten fully vaccinated
df_cohort_mom_mrna_2_shots_only_breakthrough = spark.sql(
"""
FROM mrna_2_shots AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.full_vaccination_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

# identify those that had a breakthrough infection after being fully vaccinated, but before getting boosted
df_cohort_mom_mrna_full_vaccination_breakthrough = spark.sql(
"""
FROM mrna_boosted AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.full_vaccination_date < cm.covid_test_date
  AND mrna.last_immunization_date > cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])

# join together all that received breakthrough infection after 2 shots only
df_cohort_mom_mrna_2_shots_only_breakthrough_final = df_cohort_mom_mrna_2_shots_only_breakthrough.union(df_cohort_mom_mrna_full_vaccination_breakthrough).dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_mrna_2_shots_only_breakthrough_final = df_cohort_mom_mrna_2_shots_only_breakthrough_final.withColumn('days_protected_from_covid', datediff('covid_test_date', 'full_vaccination_date'))
write_data_frame_to_sandbox(df_cohort_mom_mrna_2_shots_only_breakthrough_final, 'snp2_cohort_maternity_mrna_2_shots_only_covid_breakthrough', sandbox_db='rdp_phi_sandbox', replace=True)

# identify those with breakthrough infection after getting boosted
df_cohort_mom_mrna_boosted_breakthrough = spark.sql(
"""
FROM mrna_boosted AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.last_immunization_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name, mrna.booster_date, mrna.booster_name 
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_mrna_boosted_breakthrough = df_cohort_mom_mrna_boosted_breakthrough.withColumn('days_protected_from_covid', datediff('covid_test_date', 'full_vaccination_date'))
df_cohort_mom_mrna_boosted_breakthrough = df_cohort_mom_mrna_boosted_breakthrough.withColumn('days_protected_from_covid_boosted', datediff('covid_test_date', 'booster_date'))
write_data_frame_to_sandbox(df_cohort_mom_mrna_boosted_breakthrough, 'snp2_cohort_maternity_mrna_boosted_covid_breakthrough', sandbox_db='rdp_phi_sandbox', replace=True)

# identify those with breakthrough infection that were vaccinated, but not boosted
df_cohort_mom_vaccinated_but_not_boosted_breakthrough = spark.sql(
"""
FROM vaccinated_but_not_boosted AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.last_immunization_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_vaccinated_but_not_boosted_breakthrough = df_cohort_mom_vaccinated_but_not_boosted_breakthrough.withColumn('days_protected_from_covid', datediff('covid_test_date', 'full_vaccination_date'))
write_data_frame_to_sandbox(df_cohort_mom_vaccinated_but_not_boosted_breakthrough, 'snp2_cohort_maternity_vaccinated_but_not_boosted_covid_breakthrough', sandbox_db='rdp_phi_sandbox', replace=True)

# identify those with breakthrough infection that were vaccinated and had bivalent booster, but not boosted 
df_cohort_mom_vaccinated_bivalent_boosted_breakthrough = spark.sql(
"""
FROM vaccinated_bivalent_boosted AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.last_immunization_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_vaccinated_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.withColumn('days_protected_from_covid', datediff('covid_test_date', 'full_vaccination_date'))
df_cohort_mom_vaccinated_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.withColumn('days_protected_from_covid_bivalent_boosted', datediff('covid_test_date', 'last_immunization_date'))
write_data_frame_to_sandbox(df_cohort_mom_vaccinated_bivalent_boosted_breakthrough, 'snp2_cohort_maternity_vaccinated_bivalent_boosted_covid_breakthrough', sandbox_db='rdp_phi_sandbox', replace=True)

# identify those with breakthrough infection after getting vaccinated, boosted, and bivalent boosted
df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = spark.sql(
"""
FROM vaccinated_boosted_bivalent_boosted AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.last_immunization_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name, mrna.full_vaccination_date, mrna.full_vaccination_name, mrna.booster_date, mrna.booster_name 
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.withColumn('days_protected_from_covid', datediff('covid_test_date', 'full_vaccination_date'))
df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.withColumn('days_protected_from_covid_boosted', datediff('covid_test_date', 'booster_date'))
df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.withColumn('days_protected_from_covid_bivalent_boosted', datediff('covid_test_date', 'last_immunization_date'))
write_data_frame_to_sandbox(df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough, 'snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted_covid_breakthrough', sandbox_db='rdp_phi_sandbox', replace=True)


# identify those with breakthrough infection after 1 shot only
df_mrna_1_shot_covid_infection = spark.sql(
"""
FROM mrna_1_shot AS mrna
INNER JOIN covid_mom AS cm
ON mrna.pat_id == cm.pat_id
  AND mrna.episode_id == cm.episode_id
  AND mrna.child_episode_id == cm.child_episode_id
  AND mrna.conception_date < cm.covid_test_date
  AND mrna.ob_delivery_delivery_date > cm.covid_test_date
  AND mrna.first_immunization_date < cm.covid_test_date
SELECT cm.*, mrna.first_immunization_date, mrna.first_immunization_name, mrna.last_immunization_date, mrna.last_immunization_name
""").dropDuplicates(['pat_id', 'episode_id', 'child_episode_id'])
write_data_frame_to_sandbox(df_mrna_1_shot_covid_infection, 'snp2_cohort_maternity_mrna_1_shot_covid_infection', sandbox_db='rdp_phi_sandbox', replace=True)


df_cohort_mom_mrna_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_mrna")
df_cohort_mom_mrna_breakthrough.createOrReplaceTempView("mrna_breakthrough")
print('# Number of people fully vaccinated at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_mrna_breakthrough.count()))

df_cohort_mom_mrna_2_shots_only_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only_covid_breakthrough")
df_cohort_mom_mrna_2_shots_only_breakthrough.createOrReplaceTempView("mrna_2_shots_breakthrough")
print('# Number of people with only 2 shots (mrna) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_mrna_2_shots_only_breakthrough.count()))

df_cohort_mom_mrna_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted_covid_breakthrough")
df_cohort_mom_mrna_boosted_breakthrough.createOrReplaceTempView("mrna_boosted_breakthrough")
print('# Number of people boosted (mrna) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_mrna_boosted_breakthrough.count()))
print('\n')

df_cohort_mom_vaccinated_but_not_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted_covid_breakthrough")
print('# Number of people vaccinated but not boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count()))

df_cohort_mom_vaccinated_bivalent_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_bivalent_boosted_covid_breakthrough")
print('# Number of people vaccinated and bivalent boosted but not boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.count()))

df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted_covid_breakthrough")
print('# Number of people vaccinated, boosted, and bivalent boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.count()))
print('\n')


df_cohort_mom_moderna_2_shots_breakthrough = df_cohort_mom_mrna_2_shots_only_breakthrough.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of people with only 2 shots (moderna) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_moderna_2_shots_breakthrough.count()))

df_cohort_mom_moderna_boosted_breakthrough = df_cohort_mom_mrna_boosted_breakthrough.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of people boosted (moderna) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_moderna_boosted_breakthrough.count()))
print('\n')


df_cohort_mom_pfizer_2_shots_breakthrough = df_cohort_mom_mrna_2_shots_only_breakthrough.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of people with only 2 shots (pfizer) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_pfizer_2_shots_breakthrough.count()))

df_cohort_mom_pfizer_boosted_breakthrough = df_cohort_mom_mrna_boosted_breakthrough.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of people boosted (pfizer) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_pfizer_boosted_breakthrough.count()))
print('\n')


df_cohort_mom_jj_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_jj")
df_cohort_mom_jj_breakthrough.createOrReplaceTempView("jj_breakthrough")
print('# Number of people fully vaccinated with J&J at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_jj_breakthrough.count()))
print('\n')


df_cohort_mom_mrna_1_shot_covid_infection = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_covid_infection")
df_cohort_mom_mrna_1_shot_covid_infection.createOrReplaceTempView("mrna_1_shot_covid_infection")
print('# Number of people fully vaccinated with only 1 mRNA shot before contracting covid during pregnancy: ' + str(df_cohort_mom_mrna_1_shot_covid_infection.count()))
print('\n')


df_cohort_mom_covid_induced_immunity_covid_infection = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_unvaccinated_covid_immunity")
df_cohort_mom_covid_induced_immunity_covid_infection.createOrReplaceTempView("covid_induced_immunity_covid_infection")
print('# Number of people with covid-induced immunity prior to pregnancy with a covid infection during pregnancy: ' + str(df_cohort_mom_covid_induced_immunity_covid_infection.count()))
print('\n')


df_unvaccinated_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_unvaccinated")
print('# Number of unvaccinated pregnant women who have delivered that had covid during pregnancy: ' + str(df_unvaccinated_covid.count()))
df_unvaccinated_covid.createOrReplaceTempView("unvaccinated_mom_covid")
df_unvaccinated_covid_limited = spark.sql("SELECT * FROM unvaccinated_mom_covid AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")
print('# Number of unvaccinated pregnant people who have delivered after 1-25-21 that had covid during pregnancy: ' + str(df_unvaccinated_covid_limited .count()))
print('\n')


# evaluate basic stats and differences between cohorts
df_cohort_mom_mrna_2_shots_only_breakthrough_pandas = df_cohort_mom_mrna_2_shots_only_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas.days_protected_from_covid.max())))

df_cohort_mom_mrna_boosted_breakthrough_pandas = df_cohort_mom_mrna_boosted_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_mrna_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_mrna_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_mrna_boosted_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_mrna_boosted_breakthrough_pandas.days_protected_from_covid.max())))
print('\n')
print('# Mann Whitney U Test - Days Protected From Covid:')
print('# p-value: ' + str(run_mann_whitney_u_test(df_cohort_mom_mrna_boosted_breakthrough_pandas, df_cohort_mom_mrna_2_shots_only_breakthrough_pandas, 'days_protected_from_covid')[1]))


df_cohort_mom_moderna_2_shots_breakthrough_pandas = df_cohort_mom_moderna_2_shots_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_moderna_2_shots_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_moderna_2_shots_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_moderna_2_shots_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_moderna_2_shots_breakthrough_pandas.days_protected_from_covid.max())))


df_cohort_mom_pfizer_2_shots_breakthrough_pandas = df_cohort_mom_pfizer_2_shots_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_pfizer_2_shots_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_pfizer_2_shots_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_pfizer_2_shots_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_pfizer_2_shots_breakthrough_pandas.days_protected_from_covid.max())))

print('# Mann Whitney U Test - Days Protected From Covid:')
print('# p-value: ' + str(run_mann_whitney_u_test(df_cohort_mom_moderna_2_shots_breakthrough_pandas, df_cohort_mom_pfizer_2_shots_breakthrough_pandas, 'days_protected_from_covid')[1]))


df_cohort_mom_moderna_boosted_breakthrough_pandas = df_cohort_mom_moderna_boosted_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_moderna_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_moderna_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_moderna_boosted_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_moderna_boosted_breakthrough_pandas.days_protected_from_covid.max())))


df_cohort_mom_pfizer_boosted_breakthrough_pandas = df_cohort_mom_pfizer_boosted_breakthrough.toPandas()
print('# Median Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(statistics.median(df_cohort_mom_pfizer_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# IQR Breakthrough COVID-Postive Patients Days From Last Shot: ' + str(stats.iqr(df_cohort_mom_pfizer_boosted_breakthrough_pandas.days_protected_from_covid)))
print('# Min: ' + (str(df_cohort_mom_pfizer_boosted_breakthrough_pandas.days_protected_from_covid.min())))
print('# Max: ' + (str(df_cohort_mom_pfizer_boosted_breakthrough_pandas.days_protected_from_covid.max())))

print('# Mann Whitney U Test - Days Protected From Covid:')
print('# p-value: ' + str(run_mann_whitney_u_test(df_cohort_mom_moderna_boosted_breakthrough_pandas, df_cohort_mom_pfizer_boosted_breakthrough_pandas, 'days_protected_from_covid')[1]))



df_cohort_mom_mrna_2_shots_no_booster = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 168) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date")
print('# Number of fully vaccinated pregnant patients that delivered 6+ months after obtaining full vaccination status, but were not boosted by time of delivery: ' + str(df_cohort_mom_mrna_2_shots_no_booster.count()))

df_cohort_mom_mrna_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted AS boosted WHERE boosted.ob_delivery_delivery_date >= date_add(boosted.full_vaccination_date, 168) AND '2021-09-22' < boosted.ob_delivery_delivery_date")
print('# Number of patients that were boosted 6+ months after acheiving full vaccination status and prior to delivery: ' + str(df_cohort_mom_mrna_boosted.count()))

df_cohort_mom_vaccinated_but_not_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted_covid_breakthrough AS vbnb WHERE vbnb.days_protected_from_covid >= 182 AND vbnb.ob_delivery_delivery_date >= date_add(vbnb.full_vaccination_date, 168) AND '2021-09-22' < vbnb.ob_delivery_delivery_date")
print('# Number of fully vaccinated pregnant patients with breakthrough infections 6+ months after obtaining full vaccination status: ' + str(df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count()))

df_cohort_mom_mrna_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted_covid_breakthrough AS boosted WHERE boosted.days_protected_from_covid >= 182 AND boosted.ob_delivery_delivery_date >= date_add(boosted.full_vaccination_date, 168) AND '2021-09-22' < boosted.ob_delivery_delivery_date")
print('# Number of boosted pregnant patients with breakthrough infections 6+ months after obtaining full vaccination status: ' + str(df_cohort_mom_mrna_boosted_breakthrough.count()))

print('# p-value: ' + str(run_mann_whitney_u_test(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas[df_cohort_mom_mrna_2_shots_only_breakthrough_pandas['days_protected_from_covid'] >= 182], df_cohort_mom_mrna_boosted_breakthrough_pandas[df_cohort_mom_mrna_boosted_breakthrough_pandas['days_protected_from_covid'] >= 182], 'days_protected_from_covid')[1]))

dict_days_protected = {'Vaccinated,\nbut not Boosted\n(n=608)': list(df_cohort_mom_mrna_2_shots_only_breakthrough_pandas[df_cohort_mom_mrna_2_shots_only_breakthrough_pandas['days_protected_from_covid'] >= 182].days_protected_from_covid), 'Boosted\n(n=337)': list(df_cohort_mom_mrna_boosted_breakthrough_pandas[df_cohort_mom_mrna_boosted_breakthrough_pandas['days_protected_from_covid']>= 182].days_protected_from_covid)}

make_n_violin_box_plot(dict_days_protected)


list_colors = ['goldenrod', 'c']
create_violin_plot(dict_days_protected, 'Number of Days from\nFull Vaccination Status', list_colors)


count_2_shot_breakthrough = df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count()
dict_2_shot_breakthrough = {'Yes': count_2_shot_breakthrough, 'No': df_cohort_mom_mrna_2_shots_no_booster.count()}
count_boosted_breakthrough = df_cohort_mom_mrna_boosted_breakthrough.count()
dict_boosted_breakthrough = {'Yes': count_boosted_breakthrough, 'No': (df_cohort_mom_mrna_boosted.count()-count_boosted_breakthrough)}

conversion = dict_2_shot_breakthrough['No']/df_cohort_mom_mrna_2_shots_no_booster.count()
ci_low, ci_up = proportion_confint(dict_2_shot_breakthrough['Yes'], dict_2_shot_breakthrough['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(dict_2_shot_breakthrough['Yes']/(dict_2_shot_breakthrough['No'] + dict_2_shot_breakthrough['Yes'])))
print('95% CI: ' + str(ci_low*conversion), str(ci_up*conversion))
print('\n')

conversion = dict_boosted_breakthrough['No']/df_cohort_mom_mrna_boosted.count()
ci_low, ci_up = proportion_confint(dict_boosted_breakthrough['Yes'], dict_boosted_breakthrough['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(dict_boosted_breakthrough['Yes']/(dict_boosted_breakthrough['No'] + dict_boosted_breakthrough['Yes'])))
print('95% CI: ' + str(ci_low*conversion), str(ci_up*conversion))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_2_shot_breakthrough, dict_boosted_breakthrough))


# redefine violin box plot function
def make_n_violin_box_plot(dict_outliers):
  dict_filtered = {}
  for k, v in dict_outliers.items():
      if len(v) > 9:
          dict_filtered[k] = v
  fig, ax = plt.subplots()

  # Create a plot
  ax.violinplot(list(dict_filtered.values()))
  ax.boxplot(list(dict_filtered.values()))

  # add x-tick labels
  xticklabels = dict_filtered.keys()
  ax.set_ylabel('Number of Days\nfrom Full Vaccination Status')
  ax.set_xticks([1,2])
  ax.set_xticklabels(xticklabels, rotation = 15)
  ax.set_ylim([0, 700])
  plt.show()

print('# p-value: ' + str(run_mann_whitney_u_test(df_cohort_mom_moderna_2_shots_breakthrough_pandas.append(df_cohort_mom_moderna_boosted_breakthrough_pandas), df_cohort_mom_pfizer_2_shots_breakthrough_pandas.append(df_cohort_mom_pfizer_boosted_breakthrough_pandas), 'days_protected_from_covid')[1]))

dict_days_protected = {'Vaccinated\nModerna\n(n=446)': list(df_cohort_mom_moderna_2_shots_breakthrough_pandas.append(df_cohort_mom_moderna_boosted_breakthrough_pandas).days_protected_from_covid), 'Vaccinated\nPfizer\n(n=640)': list(df_cohort_mom_pfizer_2_shots_breakthrough_pandas.append(df_cohort_mom_pfizer_boosted_breakthrough_pandas).days_protected_from_covid)}

make_n_violin_box_plot(dict_days_protected)
list_colors = ['royalblue', 'hotpink']
create_violin_plot(dict_days_protected, 'Number of Days from\nFull Vaccination Status', list_colors)


df_cohort_maternity_vaccinated_mrna = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna")
df_cohort_maternity_moderna = df_cohort_mom_mrna_2_shots.filter(col("first_immunization_name").like("%MODERNA%"))
df_cohort_maternity_pfizer  = df_cohort_mom_mrna_2_shots.filter(col("first_immunization_name").like("%PFIZER%")) 

count_moderna_2_shots_breakthrough = df_cohort_mom_moderna_2_shots_breakthrough.count()
dict_moderna_breakthrough = {'Yes': count_moderna_2_shots_breakthrough, 'No': (df_cohort_mom_moderna_2_shots.count()-count_moderna_2_shots_breakthrough)}
count_pfizer_2_shots_breakthrough = df_cohort_mom_pfizer_2_shots_breakthrough.count()
dict_pfizer_breakthrough = {'Yes': count_pfizer_2_shots_breakthrough, 'No': (df_cohort_mom_pfizer_2_shots.count()-count_pfizer_2_shots_breakthrough)}

conversion = dict_moderna_breakthrough['No']/df_cohort_mom_moderna_2_shots.count()
ci_low, ci_up = proportion_confint(dict_moderna_breakthrough['Yes'], dict_moderna_breakthrough['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(dict_moderna_breakthrough['Yes']/(dict_moderna_breakthrough['No'] + dict_moderna_breakthrough['Yes'])))
print('95% CI: ' + str(ci_low*conversion), str(ci_up*conversion))
print('\n')

conversion = dict_pfizer_breakthrough['No']/df_cohort_mom_pfizer_2_shots.count()
ci_low, ci_up = proportion_confint(dict_pfizer_breakthrough['Yes'], dict_pfizer_breakthrough['No'], alpha=0.05, method='wilson')
print('Percentage: ' + str(dict_pfizer_breakthrough['Yes']/(dict_pfizer_breakthrough['No'] + dict_pfizer_breakthrough['Yes'])))
print('95% CI: ' + str(ci_low*conversion), str(ci_up*conversion))
print('\n')

print(calc_fishers_exact_test_not_simulated(dict_moderna_breakthrough, dict_pfizer_breakthrough))


# get percentage of people with full vaccination and booster during pregnancy
df_cohort_3_shots_during_pregnancy = spark.sql("SELECT * FROM mrna_boosted AS mrna WHERE mrna.conception_date < mrna.first_immunization_date AND mrna.ob_delivery_delivery_date > mrna.last_immunization_date")
temp = 100*df_cohort_3_shots_during_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 3 shots during pregnancy: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 3 shots during pregnancy: ' + str(df_cohort_3_shots_during_pregnancy.count()))
print('\n')


# get percentage of people with 1 shot prior to pregnancy and 2 shots during pregnancy
df_cohort_1_shot_prior_2_shots_during_pregnancy = spark.sql("SELECT * FROM mrna_boosted AS mrna WHERE mrna.conception_date > mrna.first_immunization_date AND mrna.conception_date < mrna.full_vaccination_date AND mrna.ob_delivery_delivery_date > mrna.last_immunization_date")
temp = 100*df_cohort_1_shot_prior_2_shots_during_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 1 shot prior and 2 shots during pregnancy: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 1 shot prior and 2 shots during pregnancy: ' + str(df_cohort_1_shot_prior_2_shots_during_pregnancy.count()))
print('\n')


# get percentage of people with 2 shot prior to pregnancy and 1 shots during pregnancy
df_cohort_2_shots_prior_1_shot_during_pregnancy = spark.sql("SELECT * FROM mrna_boosted AS mrna WHERE mrna.conception_date > mrna.first_immunization_date AND mrna.conception_date > mrna.full_vaccination_date AND mrna.ob_delivery_delivery_date > mrna.last_immunization_date AND mrna.conception_date < mrna.last_immunization_date")
temp = 100*df_cohort_2_shots_prior_1_shot_during_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 2 shots prior and 1 shot during pregnancy: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 2 shots prior and 1 shot during pregnancy: ' + str(df_cohort_2_shots_prior_1_shot_during_pregnancy.count()))
print('\n')


# get percentage of people with 3 shots prior to pregnancy
df_cohort_3_shots_prior_pregnancy = spark.sql("SELECT * FROM mrna_boosted AS mrna WHERE mrna.conception_date > mrna.last_immunization_date")
temp = 100*df_cohort_3_shots_prior_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 3 shots prior to pregnancy: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 3 shots prior to pregnancy: ' + str(df_cohort_3_shots_prior_pregnancy.count()))
print('\n')


# get percentage of people with 2 shots prior to pregnancy and no booster
df_cohort_2_shots_prior_pregnancy = spark.sql("SELECT * FROM mrna_2_shots AS mrna WHERE mrna.conception_date > mrna.last_immunization_date")
temp = 100*df_cohort_2_shots_prior_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 2 shots prior to pregnancy and no booster: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 2 shots prior to pregnancy and no booster: ' + str(df_cohort_2_shots_prior_pregnancy.count()))
print('\n')


# get percentage of people with 1 shot prior to and 1 shot during pregnancy and no booster
df_cohort_1_shot_prior_1_shot_during_pregnancy = spark.sql("SELECT * FROM mrna_2_shots AS mrna WHERE mrna.conception_date > mrna.first_immunization_date AND mrna.conception_date < mrna.last_immunization_date AND mrna.ob_delivery_delivery_date > mrna.last_immunization_date")
temp = 100*df_cohort_1_shot_prior_1_shot_during_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 1 shot prior and 1 shot during pregnancy and no booster: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 1 shot prior and 1 shot during pregnancy and no booster: ' + str(df_cohort_1_shot_prior_1_shot_during_pregnancy.count()))
print('\n')


# get percentage of people with 2 shots during pregnancy and no booster
df_cohort_2_shots_during_pregnancy = spark.sql("SELECT * FROM mrna_2_shots AS mrna WHERE mrna.conception_date < mrna.first_immunization_date AND mrna.ob_delivery_delivery_date > mrna.full_vaccination_date")
temp = 100*df_cohort_2_shots_during_pregnancy.count()/df_cohort_mom_mrna_vaccinated.count()
print('# Percentage of people that were fully vaccinated at delivery that received 2 shots during pregnancy and no booster: ' + str(temp) + '%')
print('# Number of people that were fully vaccinated at delivery that received 2 shots during pregnancy and no booster: ' + str(df_cohort_2_shots_during_pregnancy.count()))


# evaluate infections by dominant variant at time of positive covid test
dict_mrna_breakthrough_variant = make_variant_covid_infection_dict(df_cohort_mom_mrna_breakthrough)
dict_unvaccinated_variant = make_variant_covid_infection_dict(df_unvaccinated_covid_limited)

calc_fishers_exact_test(dict_mrna_breakthrough_variant, dict_unvaccinated_variant)


dict_boosted_breakthrough_variant = make_variant_covid_infection_dict(df_cohort_mom_mrna_boosted_breakthrough)
dict_unvaccinated_but_not_boosted_variant = make_variant_covid_infection_dict(df_cohort_mom_vaccinated_but_not_boosted_breakthrough)

calc_fishers_exact_test(dict_boosted_breakthrough_variant, dict_unvaccinated_but_not_boosted_variant)


# load df_unvaccinated
df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_unvaccinated_expanded_6").na.drop(subset=['pat_id', 'episode_id', 'child_episode_id'])
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom")
df_unvaccinated = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")

#filter other cohorts by delivery date
df_covid_induced_immunity.createOrReplaceTempView("covid_induced_immunity")
df_covid_induced_immunity = spark.sql("SELECT * FROM covid_induced_immunity AS cii WHERE '2021-01-26' <= cii.ob_delivery_delivery_date")
df_cohort_mom_jj.createOrReplaceTempView("jj")
df_cohort_mom_jj = spark.sql("SELECT * FROM jj WHERE '2021-01-26' <= jj.ob_delivery_delivery_date")
df_cohort_mom_mrna_1_shot.createOrReplaceTempView('mrna_1_shot')
df_cohort_mom_mrna_1_shot = spark.sql("SELECT * FROM mrna_1_shot WHERE '2021-01-26' <= mrna_1_shot.ob_delivery_delivery_date")

# make line histogram of data at delivery 
list_of_df_vaccination_status_at_delivery = [df_unvaccinated, df_cohort_mom_mrna_vaccinated, df_cohort_mom_mrna_boosted]
list_data = format_data_for_delivery_chart(list_of_df_vaccination_status_at_delivery)
list_colors = ['crimson', 'seagreen', 'c']
plot_vaccination_status_at_delivery_chart(list_data, list_colors)


# evaluate timing of covid infections
df_unvaccinated_covid_limited.agg({'covid_test_date': 'min'}).show()
df_unvaccinated_covid_limited.agg({'covid_test_date': 'max'}).show()

df_cohort_mom_mrna_breakthrough.agg({'covid_test_date': 'min'}).show()
df_cohort_mom_mrna_breakthrough.agg({'covid_test_date': 'max'}).show()


# redefine functions related to histograms
def calc_covid_test_date_days_from_start(list_date_covid_test):
  n_days = []
  for date_covid_test in list_date_covid_test:
    dif = (date_covid_test.date() - DATE_START).days
    n_days.append(dif)
  return n_days

    
def format_data_for_covid_infection_chart(data):
  data_final = []
  for df in data:
    df = df.toPandas()
    n_days = calc_covid_test_date_days_from_start(list(df['covid_test_date']))
    data_final.append(n_days)
  return data_final
    

def plot_covid_infection_chart(list_data, list_colors):
  n_bins = 91
  x_min, x_max = 0, 637
  x_ticks = ("Jan 26, '21", "Feb 23, '21", "Mar 23, '21", "Apr 20, '21", "May 18, '21", "Jun 15, '21", "Jul 13, '21", "Aug 10, '21", "Sep 7, '21", "Oct 5, '21", "Nov 2, '21", "Nov 30, '21", "Dec 28, '21", "Jan 25, '22", "Feb 22, '22", "Mar 22, '22", "Apr 19, '22", "May 17, '22", "Jun 14, '22", "Jul 12, '22", "Aug 9, '22", "Sep 6, '22", "Oct 4, '22") 
  c=0
  
  for data in list_data:  # plot different vaccination groups
    color = list_colors[c]
    y, binEdges = np.histogram(data, bins=n_bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c=color)
    c+=1
  
  plt.axvline(x=333, color='k', linestyle='--')
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min, x_max, 28.0), x_ticks, fontsize=12, rotation=75)
  plt.ylabel("Number of Pregnant People\nwith COVID-19 Infections")
  plt.tight_layout()
  plt.show()


# load data for covid infections in the unvaccinated cohort
df_unvaccinated_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_unvaccinated")
df_unvaccinated_covid.createOrReplaceTempView("unvaccinated_mom_covid")
df_unvaccinated_covid_limited = spark.sql("SELECT * FROM unvaccinated_mom_covid AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")

# load data for covid infections in other cohorts
df_mom_jj_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_jj")
df_mom_mrna_1_shot_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_covid_infection")
df_mom_covid_induced_immunity_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_unvaccinated_covid_immunity")
df_mom_mrna_2_shots_only_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only_covid_breakthrough")
df_mom_mrna_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted_covid_breakthrough")

# make line histogram of covid infection dates
list_of_df_vaccination_status_at_delivery = [df_unvaccinated_covid_limited, df_cohort_mom_mrna_breakthrough, df_mom_mrna_boosted_breakthrough]
list_data = format_data_for_covid_infection_chart(list_of_df_vaccination_status_at_delivery)
list_colors = ['crimson', 'seagreen', 'c']
plot_covid_infection_chart(list_data, list_colors)


# redefine functions related to plotting covid infections
def initiate_dict(x_max):
  d = {}
  c = 0
  while c < x_max:
    d[c] = 0
    c += 1
  return d


def make_delivery_date_days_from_start_dict(list_date_delivery, x_max):
  d = initiate_dict(x_max)
  for date_delivery in list_date_delivery:
    dif = (date_delivery.date() - DATE_START).days
    if dif < x_max:
      d[dif] += 1
  return d


def calc_proportion_with_covid_at_delivery(dict_delivery, dict_covid, x_max):
  list_proportion = []
  n_delivery = 0
  n_covid = 0
  c = 0
  while c < x_max:
    n_delivery += dict_delivery[c]
    n_covid += dict_covid[c]
    if c%7 == 0:
      if n_delivery > 9:
        list_proportion.append(100*n_covid/n_delivery)
      else:
        list_proportion.append(0)
      n_delivery = 0
      n_covid = 0
    c += 1
  return list_proportion

    
def format_data_for_chart(data):
  data_final = []
  x_max = 637
  for df_delivery, df_covid in data:
    df_delivery = df_delivery.toPandas()
    dict_delivery = make_delivery_date_days_from_start_dict(list(df_delivery['ob_delivery_delivery_date']), x_max)
    df_covid = df_covid.toPandas()
    dict_covid = make_delivery_date_days_from_start_dict(list(df_covid['ob_delivery_delivery_date']), x_max)
    list_proportion = calc_proportion_with_covid_at_delivery(dict_delivery, dict_covid, x_max)
    data_final.append(list_proportion)
  return data_final


def plot_covid_infection_chart(list_data, list_colors):
  y_min, y_max = 0, 100
  x_min, x_max = 0, 637
  x_ticks = ("Jan 26, '21", "Feb 23, '21", "Mar 23, '21", "Apr 20, '21", "May 18, '21", "Jun 15, '21", "Jul 13, '21", "Aug 10, '21", "Sep 7, '21", "Oct 5, '21", "Nov 2, '21", "Nov 30, '21", "Dec 28, '21", "Jan 25, '22", "Feb 22, '22", "Mar 22, '22", "Apr 19, '22", "May 17, '22", "Jun 14, '22", "Jul 12, '22", "Aug 9, '22", "Sep 6, '22", "Oct 4, '22") 
  c=0
  
  for data in list_data:  # plot different vaccination groups
    color = list_colors[c]
    plt.plot(np.arange(x_min, x_max, 7), data, '-', c=color)
    c+=1
  
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min, x_max, 28.0), x_ticks, fontsize=12, rotation=75)
  plt.ylim(y_min, y_max)
  plt.ylabel("Percentage of People at Delivery \nwith COVID-19 during Pregnancy")
  plt.tight_layout()
  plt.show()

  
def add_list_to_df(df, l, cond):
  c = 0
  for item in l:
    d = {'sub_id': c, 'y': item, 'cond': cond}
    df = df.append(d, ignore_index=True)
    c += 1
  return df
  
  
def perform_anova(l_experimental, l_control):
  df = pd.DataFrame(columns=['sub_id', 'y', 'cond'])
  df = add_list_to_df(df, l_experimental, 'experimental')
  df = add_list_to_df(df, l_control, 'control')
  aovrm = AnovaRM(df, 'y', 'sub_id', within=['cond'])
  res = aovrm.fit()
  return res


# make chart of proportion of pregnancies with covid-19 infections at time of delivery
list_of_cohort_pairs = [(df_unvaccinated_limited, df_unvaccinated_covid_limited), (df_cohort_mom_mrna_vaccinated, df_cohort_mom_mrna_breakthrough), (df_cohort_mom_mrna_boosted, df_mom_mrna_boosted_breakthrough)]
list_data = format_data_for_chart(list_of_cohort_pairs)
list_colors = ['crimson', 'seagreen', 'c']
plot_covid_infection_chart(list_data, list_colors)

# calculate p-values using a repeated measurement ANOVA test
print('Unvaccinated vs Vaccinated:')
print(perform_anova(list_data[0], list_data[1]))
print('\n')
print('P-value (Unvaccinated vs Booster):')
print(perform_anova(list_data[0], list_data[2]))
print('\n')
print('P-value (Vaccinated vs Booster):')
print(perform_anova(list_data[1], list_data[2]))
print('\n')


# evaluate gestational timing of full vaccination status
df_cohort_full_vaccination_status_during_pregnancy = spark.sql("SELECT * FROM mrna_vaccinated AS mrna WHERE date_sub(mrna.ob_delivery_delivery_date, mrna.gestational_days) < mrna.full_vaccination_date AND mrna.ob_delivery_delivery_date > mrna.full_vaccination_date").toPandas()
print('# Number of pregnant people that achieved full vaccination status during pregnancy: ' + str(len(df_cohort_full_vaccination_status_during_pregnancy)))
day_of_gestation_of_full_vaccination_status = calc_gestational_day_of_full_vaccination_status(df_cohort_full_vaccination_status_during_pregnancy)
calc_trimester_of_full_vaccination_status(day_of_gestation_of_full_vaccination_status)
plot_histogram(day_of_gestation_of_full_vaccination_status, 'seagreen', 'COVID-19 Full Vaccination Acheived')


# evaluate gestational timing of receiving a booster shot
df_cohort_mom_mrna_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted AS mrna WHERE date_sub(mrna.ob_delivery_delivery_date, mrna.gestational_days) < mrna.booster_date AND mrna.ob_delivery_delivery_date > mrna.booster_date").toPandas()
print('# Number of pregnant people that achieved full vaccination status during pregnancy: ' + str(len(df_cohort_mom_mrna_boosted)))
day_of_gestation_of_booster_shot = calc_gestational_day_of_booster_shot(df_cohort_mom_mrna_boosted)
calc_trimester_of_full_vaccination_status(day_of_gestation_of_booster_shot)
plot_histogram(day_of_gestation_of_booster_shot, 'c', 'Booster Shot Was Received')

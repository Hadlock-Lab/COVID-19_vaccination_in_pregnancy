# Author: Samantha Piekos
# Date: 11/9/22


# load the environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics

from datetime import date, datetime, timedelta
from dateutil.relativedelta import *
from pyspark.sql.functions import *
from pyspark.sql.functions import col, lit, unix_timestamp
from scipy import stats
from sklearn.impute import SimpleImputer

dbutils.library.installPyPI("lifelines")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

%matplotlib inline


# declare universal variables
DATE_START_CONCEPTION = date(2020, 10, 22)
DATE_START_FULL_VACCINATION = date(2021, 1, 18)
DATE_CUTOFF = datetime.today()


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.general_utilities
import COVID19_vaccination_in_pregnancy.utilities.sars_cov_2_cohort_functions


# define functions
def determine_birth_date(delivery_date, working_delivery_date, lmp, gestational_days):
  gestation = 280
  date = False
  if gestational_days > 0:
    gestation = gestational_days
  if pd.isnull(delivery_date):
    if pd.isnull(working_delivery_date):
      if pd.isnull(lmp):
        return(date, gestation)
      else:
        date = lmp + gestation
    else:
      date = working_delivery_date
  else:
    date = delivery_date
  return(date, gestation)


def calculate_days_protected(covid_test_date, start_date, days_protected):
  if covid_test_date > start_date:
      days_protected_updated = int((covid_test_date - start_date)/np.timedelta64(1, 'D'))
      if 0 <= days_protected_updated < days_protected:
        return(days_protected_updated, True)
  return(days_protected, False)


def determine_days_protected(row, df_covid_mom, start_date):
  days_protected = 182  # 6 months
  if row['pat_id'] in list(df_covid_mom['pat_id']):
    covid_test_date = df_covid_mom.loc[df_covid_mom['pat_id'] == row['pat_id']]['covid_test_date'].unique()[0]
    return(calculate_days_protected(covid_test_date, start_date, days_protected))
  return(days_protected, False)


def define_full_vaccination_pregnant_cohort(df, df_covid_mom):
  df['conception_date'] = pd.to_datetime(df['conception_date'])
  df_full_vaccination = pd.DataFrame(columns=['pat_id', 'instance', 'delivery_date', 'days_protected_from_covid', 'breakthrough_infection', 'days_since_vaccination', 'gestational_age_at_full_vaccination'])
  df_filtered = pd.DataFrame(columns=df.columns)
  for index, row in df.iterrows():
    death_event = 0
    delivery_date, gestational_days = determine_birth_date(row['ob_delivery_delivery_date'], row['working_delivery_date'], row['lmp'], row['gestational_days'])
    full_vaccination_date = row['full_vaccination_date']
    conception_date = row['conception_date']
    gestational_age_at_full_vaccination = int((full_vaccination_date - conception_date)/np.timedelta64(1, 'D'))
    #if delivery_date == False:
    #  continue
    days_protected, breakthrough_infection = determine_days_protected(row, df_covid_mom, pd.to_datetime(full_vaccination_date))
    days_since_vaccination = (DATE_CUTOFF - pd.to_datetime(full_vaccination_date)).days
    if breakthrough_infection:
      death_event = 1
    #if (full_vaccination_date + timedelta(days=days_protected)) <= delivery_date and (full_vaccination_date + timedelta(days=days_protected)) < DATE_CUTOFF and conception_date <= full_vaccination_date:
    df_filtered = df_filtered.append(row)
    df_full_vaccination = df_full_vaccination.append({'pat_id': row['pat_id'],\
                                                      'instance': row['instance'],\
                                                      'delivery_date': delivery_date,\
                                                      'days_protected_from_covid': days_protected,\
                                                      'breakthrough_infection': death_event,\
                                                      'days_since_vaccination': days_since_vaccination,\
                                                      'gestational_age_at_full_vaccination': gestational_age_at_full_vaccination},\
                                                      ignore_index=True)
  df_final = pd.merge(df_filtered, df_full_vaccination, on=['pat_id', 'instance'])
  return df_final


def make_conception_day_dicts(df):
  d = {}
  d_2 = {}
  for index, row in df.iterrows():
    conception_date, full_vaccination_date = row['conception_date'], row['full_vaccination_date']
    if conception_date not in d:
      d[conception_date] = 1
      d_2[conception_date] = [full_vaccination_date]
    else:
      d[conception_date] += 1
      d_2[conception_date].append(full_vaccination_date)
  return d, d_2


def calculate_conception_date(df):
  df['conception_date'] = np.nan
  for index, row in df.iterrows():
    df.at[index, 'conception_date'] = row['ob_delivery_delivery_date'] - timedelta(days=row['gestational_days'])
  return df


'''
def randomly_add_matched_full_vaccination_date(df, l):
  count = 0
  for index, row in df.iterrows():
    df.at[index,'matched_patient_full_vaccination_date'] = l[count]
    count += 1
  return df


def select_random_unvaccinated_date_matched(df, dict_conception_date_count, dict_full_vaccination_date):
  for key, value in dict_conception_date_count.items():
    df_matched = pd.DataFrame(columns=df.columns)
    df['matched_patient_full_vaccination_date'] = np.nan
    df_temp = df.loc[df['conception_date'] == key]
    df_temp_random = df.sample(n=value, random_state=611)
    df_temp_random = randomly_add_matched_full_vaccination_date(df_temp_random, dict_full_vaccination_date[key])
    df_matched = df_matched.append(df_temp_random, ignore_index=True)
  print(df_matched.head())
  return df_matched
'''


def get_matching_conception_dates(df, conception_date):
  df_temp = df.loc[df['conception_date'] == conception_date]
  count = 1
  while len(df_temp) == 0:
    df_temp_0 = df.loc[df['conception_date'] == conception_date - timedelta(days=count)]
    df_temp_1 = df.loc[df['conception_date'] == conception_date + timedelta(days=count)]
    df_temp = df_temp_0.append(df_temp_1, ignore_index=True)
  return df_temp


def select_matched_timeframe(df, full_vaccination_date):
  while len(df) > 0:
    df_temp = df.sample(n=1, random_state=611)
    index = df_temp.index[0]
    delivery_date = df_temp.at[index, 'ob_delivery_delivery_date']
    conception_date = df_temp.at[index, 'conception_date']
    if delivery_date >= full_vaccination_date + timedelta(days=182) and conception_date <= full_vaccination_date:
      df_temp.at[index, 'full_vaccination_date'] = full_vaccination_date
      return df_temp
    else:
      df = df.drop(index=df_temp.index[0])
  return False


def select_random_unvaccinated_patients(df, df_vaccinated):
  print('Start Unvaccinated Patient Selection...')
  df_final = pd.DataFrame(columns=df.columns)
  df_final['full_vaccination_date'] = np.nan
  count_patient = 0
  for index, row in df_vaccinated.iterrows():
    count_days = 0
    random_matched = False
    conception_date = row['conception_date']
    full_vaccination_date = row['full_vaccination_date']
    while random_matched is False:
      conception_date = conception_date + timedelta(days=count_days)
      df_temp = get_matching_conception_dates(df, conception_date)
      random_matched = select_matched_timeframe(df_temp, full_vaccination_date)
      count_days += 1
    df_final = df_final.append(random_matched)
    if random_matched.index[0] in df.index:
      df = df.drop(index=random_matched.index[0])
    count_patient += 1
    if count_patient%1000 == 0:
      print('Matched', str(count_patient), 'Unvaccinated Patients...')
  print('Matched All Unvaccinated Patients!')
  return df_final


def define_unvaccinated_pregnant_cohort(df, df_covid_mom, df_vaccinated):
  dict_conception_date_count, dict_full_vaccination_date = make_conception_day_dicts(df_vaccinated)
  df = calculate_conception_date(df)
  # df_matched = select_random_unvaccinated_date_matched(df, dict_conception_date_count, dict_full_vaccination_date)
  df_matched = select_random_unvaccinated_patients(df, df_vaccinated)
  print('Successfully Chronologically Matched Unvaccinated Patients!')
  df_unvaccinated_matched = pd.DataFrame(columns=['pat_id', 'instance', 'delivery_date', 'days_protected_from_covid', 'breakthrough_infection', 'days_since_vaccination', 'gestational_age_at_full_vaccination'])
  df_filtered = pd.DataFrame(columns=df.columns)
  for index, row in df_matched.iterrows():
    death_event = 0
    delivery_date, gestational_days = determine_birth_date(row['ob_delivery_delivery_date'], row['working_delivery_date'], row['lmp'], row['gestational_days'])
    #if delivery_date == False:
    #  continue
    full_vaccination_date = row['full_vaccination_date']
    conception_date = row['conception_date']
    days_protected, breakthrough_infection = determine_days_protected(row, df_covid_mom, pd.to_datetime(full_vaccination_date))
    days_since_vaccination = (DATE_CUTOFF - pd.to_datetime(full_vaccination_date)).days
    gestational_age_at_full_vaccination = int((full_vaccination_date - conception_date)/np.timedelta64(1, 'D'))
    if breakthrough_infection:
      death_event = 1
    #if (full_vaccination_date + timedelta(days=days_protected)) <= delivery_date and (full_vaccination_date + timedelta(days=days_protected)) < DATE_CUTOFF and conception_date <= full_vaccination_date:
    df_filtered = df_filtered.append(row)
    df_unvaccinated_matched = df_unvaccinated_matched.append({'pat_id': row['pat_id'],\
                                                      'instance': row['instance'],\
                                                      'delivery_date': delivery_date,\
                                                      'days_protected_from_covid': days_protected,\
                                                      'breakthrough_infection': death_event,\
                                                      'days_since_vaccination': days_since_vaccination,\
                                                      'gestational_age_at_full_vaccination': gestational_age_at_full_vaccination},\
                                                      ignore_index=True)
  df_final = pd.merge(df_filtered, df_unvaccinated_matched, on=['pat_id', 'instance'])
  print('Done Creating Time Matched Unvaccinated Pregnant Patient Cohort!')
  return df_final


# load cohorts
df_cohort_covid_maternity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_test_data").drop('pat_enc_csn_id_collect_list', 'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list', 'results_category', 'first_ordering_datetime').toPandas().sort_index()

df_vaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna AS v WHERE v.full_vaccination_date + interval '6' month <= v.ob_delivery_delivery_date AND v.full_vaccination_date >= v.conception_date").drop('all_immunization_dates')
df_vaccinated.createOrReplaceTempView("mrna_2_shots")
print('# Number of people that were vaccinated at least 6 months prior to delivery: ' + str(df_vaccinated.count()))
df_vaccinated = df_vaccinated.toPandas()

df_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated")
print('# Number of unvaccinated pregnant women who have delivered: ' + str(df_unvaccinated.count()))
df_unvaccinated.createOrReplaceTempView("unvaccinated_mom") 
df_unvaccinated_limited = spark.sql("SELECT * FROM unvaccinated_mom AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")
print('# Number of unvaccinated pregnant people who have delivered after 1-25-21: ' + str(df_unvaccinated_limited.count()))
df_unvaccinated_limited = df_unvaccinated_limited.toPandas()
print('\n')


# define cohorts for kaplan meier analysis
df_cohort_maternity_vaccinated_final = define_full_vaccination_pregnant_cohort(df_vaccinated, df_cohort_covid_maternity)
df_cohort_maternity_unvaccinated_final = define_unvaccinated_pregnant_cohort(df_unvaccinated_limited, df_cohort_covid_maternity, df_cohort_maternity_vaccinated_final)

print('# The number of patients fully vaccinated (mRNA) at least 6 months prior to delivery: ' + str(len(df_cohort_maternity_vaccinated_final)))
print('# The number of time matched unvaccinated patients: ' + str(len(df_cohort_maternity_unvaccinated_final)))

print('# The number of breakthrough infections observed within 182 days of full vaccination (mRNA): ' + str(len((df_cohort_maternity_vaccinated_final[df_cohort_maternity_vaccinated_final['days_protected_from_covid'] < 182]['days_protected_from_covid']))))
print('# The number of infections observed within 182 days amoung unvaccinated pregnant women: ' + str(len((df_cohort_maternity_unvaccinated_final[df_cohort_maternity_unvaccinated_final['days_protected_from_covid'] < 182]['days_protected_from_covid']))))

print(df_cohort_maternity_vaccinated_final.full_vaccination_date.min())
print(df_cohort_maternity_vaccinated_final.full_vaccination_date.max())

print(df_cohort_maternity_unvaccinated_final.full_vaccination_date.min())
print(df_cohort_maternity_unvaccinated_final.full_vaccination_date.max())


# define kaplan meier curve plotting function
def plot_kaplan_meier_curve(df_vaccinated, df_unvaccinated, ylim=0, reverse=False):
  kmf = KaplanMeierFitter()
  ax = plt.subplot(111)
  t = np.linspace(0, 182, 182)
  plt.ylim((ylim,1))

  kmf.fit(df_vaccinated['days_protected_from_covid'], event_observed=df_vaccinated['breakthrough_infection'], timeline=t, label="Vaccinated")
  kmf.plot_survival_function(ax=ax, color='seagreen')
  
  print('Vaccinated vs Unvaccinated')
  results = logrank_test(df_vaccinated['days_protected_from_covid'], df_unvaccinated['days_protected_from_covid'], df_vaccinated['breakthrough_infection'], df_unvaccinated['breakthrough_infection'])
  print(results.print_summary())
  print('\n')
  
  kmf.fit(df_unvaccinated['days_protected_from_covid'], event_observed=df_unvaccinated['breakthrough_infection'], timeline=t, label="Unvaccinated")
  if reverse:
    ax = kmf.plot_survival_function(ax=ax, color="crimson")
    ax.invert_yaxis()
    plt.legend(loc='upper left')
  else:
    kmf.plot_survival_function(ax=ax, color="crimson")
    plt.legend(loc='lower left')

  plt.title("Time Protected from COVID-19 Infection")
  plt.show()


# plot graphs
plot_kaplan_meier_curve(df_cohort_maternity_vaccinated_final, df_cohort_maternity_unvaccinated_final)
plot_kaplan_meier_curve(df_cohort_maternity_vaccinated_final, df_cohort_maternity_unvaccinated_final, ylim=0.95, reverse=True)


# substratify and plot moderna vs pfizer kaplan meier curves
# define function
def plot_kaplan_meier_curve_2(df_vaccinated_moderna, df_vaccinated_pfizer, df_unvaccinated, ylim=0, reverse=False):
  kmf = KaplanMeierFitter()
  ax = plt.subplot(111)
  t = np.linspace(0, 182, 182)
  plt.ylim((ylim,1))

  kmf.fit(df_vaccinated_moderna['days_protected_from_covid'], event_observed=df_vaccinated_moderna['breakthrough_infection'], timeline=t, label="Vaccinated (Moderna)")
  kmf.plot_survival_function(ax=ax, color='royalblue')
  
  print('Vaccinated (Moderna) vs Unvaccinated')
  results = logrank_test(df_vaccinated_moderna['days_protected_from_covid'], df_unvaccinated['days_protected_from_covid'], df_vaccinated_moderna['breakthrough_infection'], df_unvaccinated['breakthrough_infection'])
  print(results.print_summary())
  print('\n')
  
  kmf.fit(df_vaccinated_pfizer['days_protected_from_covid'], event_observed=df_vaccinated_pfizer['breakthrough_infection'], timeline=t, label="Vaccinated (Pfizer)")
  kmf.plot_survival_function(ax=ax, color='hotpink')
  
  print('Vaccinated (Pfizer) vs Unvaccinated')
  results = logrank_test(df_vaccinated_pfizer['days_protected_from_covid'], df_unvaccinated['days_protected_from_covid'], df_vaccinated_pfizer['breakthrough_infection'], df_unvaccinated['breakthrough_infection'])
  print(results.print_summary())
  print('\n')
  
  print('Vaccinated (Moderna) vs Vaccinated (Pfizer)')
  results = logrank_test(df_vaccinated_moderna['days_protected_from_covid'], df_vaccinated_pfizer['days_protected_from_covid'], df_vaccinated_moderna['breakthrough_infection'], df_vaccinated_pfizer['breakthrough_infection'])
  print(results.print_summary())
  print('\n')
  
  kmf.fit(df_unvaccinated['days_protected_from_covid'], event_observed=df_unvaccinated['breakthrough_infection'], timeline=t, label="Unvaccinated")
  
  if reverse:
    ax = kmf.plot_survival_function(ax=ax, color="crimson")
    ax.invert_yaxis()
    plt.legend(loc='upper left')
  else:
    kmf.plot_survival_function(ax=ax, color="crimson")
    plt.legend(loc='lower left')

  plt.title("Time Protected from COVID-19 Infection")
  plt.show()


# plot graph
df_cohort_maternity_vaccinated_final_moderna = df_cohort_maternity_vaccinated_final[df_cohort_maternity_vaccinated_final['full_vaccination_name'].str.contains('MODERNA')]
df_cohort_maternity_vaccinated_final_pfizer = df_cohort_maternity_vaccinated_final[df_cohort_maternity_vaccinated_final['full_vaccination_name'].str.contains('PFIZER')]

print('# Number of patients vaccinated with moderna:', str(len(df_cohort_maternity_vaccinated_final_moderna)))
print('# Number of patients vaccinated with pfizer:', str(len(df_cohort_maternity_vaccinated_final_pfizer)))
print('# Number of total patients vaccinated with an mRNA vaccine', str(len(df_cohort_maternity_vaccinated_final)))
print('\n\n')

plot_kaplan_meier_curve_2(df_cohort_maternity_vaccinated_final_moderna, df_cohort_maternity_vaccinated_final_pfizer, df_cohort_maternity_unvaccinated_final)

plot_kaplan_meier_curve_2(df_cohort_maternity_vaccinated_final_moderna, df_cohort_maternity_vaccinated_final_pfizer, df_cohort_maternity_unvaccinated_final, ylim=0.95, reverse=True)


# evaluate vaccination and conception dates
# print earliest and latest conception dates considered for vaccinated patients
print(df_cohort_maternity_vaccinated_final.conception_date.min())
print(df_cohort_maternity_vaccinated_final.conception_date.max())

# print earliest and latest conception dates considered for unvaccinated matched patients
print(df_cohort_maternity_unvaccinated_final.conception_date.min())
print(df_cohort_maternity_unvaccinated_final.conception_date.max())

# print earliest and latest full vaccination considered for patients
print(df_cohort_maternity_vaccinated_final.full_vaccination_date.min())
print(df_cohort_maternity_vaccinated_final.full_vaccination_date.max())


# define functions for conception and vaccination histogram charts
def calc_days_from_start(list_dates, start):
  n_days = []
  for date in list_dates:
    dif = (date.date() - start).days
    n_days.append(dif)
  return n_days

    
def format_data_for_chart(data, col, start):
  data_final = []
  for df in data:
    n_days = calc_days_from_start(list(df[col]), start)
    data_final.append(n_days)
  return data_final
  
  
def plot_covid_vaccination_chart(list_data, list_colors):
  n_bins = 62
  x_min, x_max = 0, 434
  x_ticks = ("Jan 18, '21", "Feb 15, '21", "Mar 15, '21", "Apr 12, '21", "May 10, '21", "Jun 7, '21", "Jul 5, '21", "Aug 2, '21", "Aug 31, '21", "Sep 27, '21", "Oct 25, '21", "Nov 22, '21", "Dec 20, '21", "Jan 17, '22", "Feb 14, '22", "Mar 14, '22")
  # "Apr 11, '22", "May 9, '22", "Jun 6, '22"
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
  plt.ylabel("Number of Pregnant People Reaching\nFull Vaccination Status")
  plt.tight_layout()
  plt.show()
  
  
def plot_conception_chart(list_data, list_colors):
  n_bins = 66
  x_min, x_max = 0, 462
  x_ticks = ("Oct 22, '20", "Nov 19, '20", "Dec 17, '20", "Jan 14, '21", "Feb 11, '21", "Mar 11, '21", "Apr 8, '21", "May 6, '21", "Jun 3, '21", "Jul 1, '21", "Jul 29, '21", "Aug 26, '21", "Sep 23, '21", "Oct 21, '21", "Nov 18, '21", "Dec 16, '21", "Jan 13, '22")
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
  plt.ylabel("Number of Conceptions")
  plt.tight_layout()
  plt.show()


  # make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_unvaccinated_final, df_cohort_maternity_vaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'full_vaccination_date', DATE_START_FULL_VACCINATION)
list_colors = ['crimson', 'seagreen']
plot_covid_vaccination_chart(list_data, list_colors)


# make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_unvaccinated_final, df_cohort_maternity_vaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'conception_date', DATE_START_CONCEPTION)
list_colors = ['crimson', 'seagreen']
plot_conception_chart(list_data, list_colors)


# make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_unvaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'conception_date', DATE_START_CONCEPTION)
list_colors = ['crimson']
plot_conception_chart(list_data, list_colors)
# load the environment
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
dbutils.library.installPyPI("lifelines")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats


# Set Unviersal Variables
DATE_CUTOFF = datetime.today()
DATE_START_CONCEPTION = date(2021, 4, 14)
DATE_START_FULL_VACCINATION = date(2021, 4, 16)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.cohort_COVID_pregnancy_functions
import COVID19_vaccination_in_pregnancy.utilities.general


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


def define_booster_pregnant_cohort(df, df_covid_mom):
  df['conception_date'] = pd.to_datetime(df['conception_date'])
  df_booster = pd.DataFrame(columns=['pat_id', 'instance', 'delivery_date', 'days_protected_from_covid', 'breakthrough_infection', 'days_since_vaccination', 'gestational_age_at_full_vaccination'])
  df_filtered = pd.DataFrame(columns=df.columns)
  for index, row in df.iterrows():
    death_event = 0
    delivery_date, gestational_days = determine_birth_date(row['ob_delivery_delivery_date'], row['working_delivery_date'], row['lmp'], row['gestational_days'])
    booster_date = row['booster_date']
    conception_date = row['conception_date']
    if conception_date < DATE_START_CONCEPTION:
      continue
    gestational_age_at_full_vaccination = int((booster_date - conception_date)/np.timedelta64(1, 'D'))
    days_protected, breakthrough_infection = determine_days_protected(row, df_covid_mom, pd.to_datetime(booster_date))
    days_since_vaccination = (DATE_CUTOFF - pd.to_datetime(booster_date)).days
    if breakthrough_infection:
      death_event = 1
    df_filtered = df_filtered.append(row)
    df_booster = df_booster.append({'pat_id': row['pat_id'],\
                                                      'instance': row['instance'],\
                                                      'delivery_date': delivery_date,\
                                                      'days_protected_from_covid': days_protected,\
                                                      'breakthrough_infection': death_event,\
                                                      'days_since_vaccination': days_since_vaccination,\
                                                      'gestational_age_at_full_vaccination': gestational_age_at_full_vaccination},\
                                                      ignore_index=True)
  df_final = pd.merge(df_filtered, df_booster, on=['pat_id', 'instance'])
  print('# Number of boosted pregnant people in final cohort: ' + str(len(df_final)))
  return df_final


def make_conception_day_dicts(df):
  d = {}
  d_2 = {}
  for index, row in df.iterrows():
    conception_date, booster_date = row['conception_date'], row['booster_date']
    if conception_date not in d:
      d[conception_date] = 1
      d_2[conception_date] = [booster_date]
    else:
      d[conception_date] += 1
      d_2[conception_date].append(booster_date)
  return d, d_2


def calculate_conception_date(df):
  df['conception_date'] = np.nan
  for index, row in df.iterrows():
    df.at[index, 'conception_date'] = row['ob_delivery_delivery_date'] - timedelta(days=row['gestational_days'])
  return df


def get_matching_conception_dates(df, conception_date):
  df_temp = df.loc[df['conception_date'] == conception_date]
  count = 1
  while len(df_temp) == 0:
    df_temp_0 = df.loc[df['conception_date'] == conception_date - timedelta(days=count)]
    df_temp_1 = df.loc[df['conception_date'] == conception_date + timedelta(days=count)]
    df_temp = df_temp_0.append(df_temp_1, ignore_index=True)
  return df_temp


def select_matched_timeframe(df, booster_date):
  while len(df) > 0:
    df_temp = df.sample(n=1, random_state=611)
    index = df_temp.index[0]
    delivery_date = df_temp.at[index, 'ob_delivery_delivery_date']
    conception_date = df_temp.at[index, 'conception_date']
    if delivery_date >= booster_date + timedelta(days=182) and conception_date <= booster_date:
      df_temp.at[index, 'booster_date'] = booster_date
      #return(df_temp, False)
      return df_temp
    else:
      df = df.drop(index=df_temp.index[0])
  #return(False, True)
  return False


def select_random_unvaccinated_patients(df, df_vaccinated):
  print('Start vaccinated, but not boosted patient selection...')
  df_final = pd.DataFrame(columns=df.columns)
  df_final['booster_date'] = np.nan
  count_patient = 0
  for index, row in df_vaccinated.iterrows():
    count_days = 0
    random_matched = False
    #none_selected = True
    conception_date = row['conception_date']
    booster_date = row['booster_date']
    while random_matched is False:
      conception_date = conception_date + timedelta(days=count_days)
      df_temp = get_matching_conception_dates(df, conception_date)
      random_matched = select_matched_timeframe(df_temp, booster_date)
      # random_matched, none_selected = select_matched_timeframe(df_temp, booster_date)
      #if none_selected:
        #conception_date = conception_date - timedelta(days=count_days)
        #df_temp = get_matching_conception_dates(df, conception_date)
        #random_matched, none_selected = select_matched_timeframe(df_temp, booster_date)
      count_days += 1
    df_final = df_final.append(random_matched)
    if random_matched.index[0] in df.index:
      df = df.drop(index=random_matched.index[0])
    count_patient += 1
    if count_patient%1000 == 0:
      print('Matched', str(count_patient), 'vaccinated, but not boosted patients...')
  print('Matched all vaccinated, but not boosted patients!')
  return df_final


def define_unvaccinated_pregnant_cohort(df, df_covid_mom, df_vaccinated):
  dict_conception_date_count, dict_booster_date = make_conception_day_dicts(df_vaccinated)
  df = calculate_conception_date(df)
  df_matched = select_random_unvaccinated_patients(df, df_vaccinated)
  print('Successfully Chronologically Matched Unvaccinated Patients!')
  df_unvaccinated_matched = pd.DataFrame(columns=['pat_id', 'instance', 'delivery_date', 'days_protected_from_covid', 'breakthrough_infection', 'days_since_vaccination', 'gestational_age_at_full_vaccination'])
  df_filtered = pd.DataFrame(columns=df.columns)
  for index, row in df_matched.iterrows():
    death_event = 0
    delivery_date, gestational_days = determine_birth_date(row['ob_delivery_delivery_date'], row['working_delivery_date'], row['lmp'], row['gestational_days'])
    booster_date = row['booster_date']
    conception_date = row['conception_date']
    gestational_age_at_full_vaccination = int((booster_date - conception_date)/np.timedelta64(1, 'D'))
    days_protected, breakthrough_infection = determine_days_protected(row, df_covid_mom, pd.to_datetime(booster_date))
    days_since_vaccination = (DATE_CUTOFF - pd.to_datetime(booster_date)).days
    if breakthrough_infection:
      death_event = 1
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
  print('# Number of unvaccinated matched people meeting conditions: ' + str(len(df_final)))
  return df_final


def plot_kaplan_meier_curve(df_vaccinated, df_unvaccinated, ylim=0, reverse=False):
  kmf = KaplanMeierFitter()
  ax = plt.subplot(111)
  t = np.linspace(0, 182, 182)
  plt.ylim((ylim,1))

  kmf.fit(df_vaccinated['days_protected_from_covid'], event_observed=df_vaccinated['breakthrough_infection'], timeline=t, label="Boosted")
  kmf.plot_survival_function(ax=ax, color='c')
  
  print('Boosted vs Vaccinated, But Not Boosted')
  results = logrank_test(df_vaccinated['days_protected_from_covid'], df_unvaccinated['days_protected_from_covid'], df_vaccinated['breakthrough_infection'], df_unvaccinated['breakthrough_infection'])
  print(results.print_summary())
  print('\n')
  
  kmf.fit(df_unvaccinated['days_protected_from_covid'], event_observed=df_unvaccinated['breakthrough_infection'], timeline=t, label="Vaccinated, But Not Boosted")
  if reverse:
    ax = kmf.plot_survival_function(ax=ax, color="goldenrod")
    ax.invert_yaxis()
    plt.legend(loc='upper left')
  else:
    kmf.plot_survival_function(ax=ax, color="goldenrod")
    plt.legend(loc='lower left')

  plt.title("Time Protected from COVID-19 Infection")
  plt.show()


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
    ax = kmf.plot_survival_function(ax=ax, color="goldenrod")
    ax.invert_yaxis()
    plt.legend(loc='upper left')
  else:
    kmf.plot_survival_function(ax=ax, color="goldenrod")
    plt.legend(loc='lower left')

  plt.title("Time Protected from COVID-19 Infection")
  plt.show()


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
  n_bins = 52
  x_min, x_max = 0, 364
  x_ticks = ("Apr 16, '21", "May 14, '21", "Jun 11, '21", "Jul 9, '21", "Aug 6, '21", "Sep 3, '21", "Oct 1, '21", "Oct 29, '21", "Nov 26, '21", "Dec 24, '21", "Jan 22, '22", "Feb 18, '22", "Mar 18, '22", "Apr 15, '22")
  c=0
  
  for data in list_data:  # plot different vaccination groups
    color = list_colors[c]
    y, binEdges = np.histogram(data, bins=n_bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c=color)
    c+=1
  
  plt.xlim(x_min, x_max)
  plt.minorticks_on()
  plt.xticks(np.arange(x_min, x_max+28, 28.0), x_ticks, fontsize=12, rotation=75)
  plt.ylabel("Number of Pregnant People\nReceiving 3rd Booster Shot")
  plt.tight_layout()
  plt.show()
  
  
def plot_conception_chart(list_data, list_colors):
  n_bins = 47
  x_min, x_max = 0, 329
  x_ticks = ("Apr 14, '21", "May 12, '21", "Jun 9, '21", "Jul 7, '21", "Aug 4, '21", "Sep 1, '21", "Sep 29, '21", "Oct 27, '21", "Nov 24, '21", "Dec 22, '21", "Jan 19, '22", "Feb 16, '22")
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


# load cohorts
df_cohort_covid_maternity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_test_data").drop('pat_enc_csn_id_collect_list', 'ordering_datetime_collect_list', 'observation_datetime_collect_list', 'result_short_collect_list', 'flagged_as_collect_list', 'age_at_order_dt_collect_list', 'order_name_collect_list', 'results_category', 'first_ordering_datetime').toPandas().sort_index()

df_booster = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted AS boosted WHERE boosted.ob_delivery_delivery_date >= date_add(boosted.booster_date, 168) AND '2021-09-22' < boosted.ob_delivery_delivery_date AND boosted.booster_date + interval '6' month <= boosted.ob_delivery_delivery_date AND boosted.conception_date <= boosted.booster_date")
print('# Number of people boosted and delivered after 9-22-21: ' + str(df_booster.count()))
df_booster = df_booster.toPandas()

df_unboosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_2_shots_only AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 351) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date")
print('# Number of vaccinated, but not boosted and delivered after 9-22-21: ' + str(df_unboosted.count()))
df_unboosted = df_unboosted.toPandas()
print('\n')


# print earliest and latest conception dates considered for boosted patients
print(df_booster.conception_date.min())
print(df_booster.conception_date.max())


# print earliest and latest conception dates considered for vaccinated, but not boosted patients
print(df_unboosted.conception_date.min())
print(df_unboosted.conception_date.max())


# define cohorts
df_cohort_maternity_vaccinated_final = define_booster_pregnant_cohort(df_booster, df_cohort_covid_maternity)
df_cohort_maternity_unvaccinated_final = define_unvaccinated_pregnant_cohort(df_unboosted, df_cohort_covid_maternity, df_cohort_maternity_vaccinated_final)
print('# The number of patients boosted at least 6 months prior to delivery: ' + str(len(df_cohort_maternity_vaccinated_final)))
print('# The number of time matched vaccinated, but not boosted patients: ' + str(len(df_cohort_maternity_unvaccinated_final)))
print('# The number of breakthrough infections observed within 182 days of receiving the booster: ' + str(len((df_cohort_maternity_vaccinated_final[df_cohort_maternity_vaccinated_final['days_protected_from_covid'] < 182]['days_protected_from_covid']))))
print('# The number of infections observed within 182 days amoung vaccinated, but not boosted matched pregnant people: ' + str(len((df_cohort_maternity_unvaccinated_final[df_cohort_maternity_unvaccinated_final['days_protected_from_covid'] < 182]['days_protected_from_covid']))))
print(df_cohort_maternity_vaccinated_final.booster_date.min())
print(df_cohort_maternity_vaccinated_final.booster_date.max())
print(df_cohort_maternity_unvaccinated_final.booster_date.min())
print(df_cohort_maternity_unvaccinated_final.booster_date.max())


# plot kaplan meier curves
plot_kaplan_meier_curve(df_cohort_maternity_vaccinated_final, df_cohort_maternity_unvaccinated_final)
plot_kaplan_meier_curve(df_cohort_maternity_vaccinated_final, df_cohort_maternity_unvaccinated_final, ylim=0.95)
plot_kaplan_meier_curve(df_cohort_maternity_vaccinated_final, df_cohort_maternity_unvaccinated_final, ylim=0.95, reverse=True)


# evaluate timeperiod of analyses
# print earliest and latest conception dates considered for vaccinated patients
print(df_cohort_maternity_vaccinated_final.conception_date.min())
print(df_cohort_maternity_vaccinated_final.conception_date.max())

# print earliest and latest conception dates considered for unvaccinated matched patients
print(df_cohort_maternity_unvaccinated_final.conception_date.min())
print(df_cohort_maternity_unvaccinated_final.conception_date.max())

# print earliest and latest full vaccination considered for patients
print(df_cohort_maternity_vaccinated_final.booster_date.min())
print(df_cohort_maternity_vaccinated_final.booster_date.max())


# make descriptive plots related to the analysis
# make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_vaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'booster_date', DATE_START_FULL_VACCINATION)
list_colors = ['c']
plot_covid_vaccination_chart(list_data, list_colors)

# make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_unvaccinated_final, df_cohort_maternity_vaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'conception_date', DATE_START_CONCEPTION)
list_colors = ['goldenrod', 'c']
plot_conception_chart(list_data, list_colors)

# make line histogram of covid vaccination dates
list_of_df_vaccination = [df_cohort_maternity_unvaccinated_final]
list_data = format_data_for_chart(list_of_df_vaccination, 'conception_date', DATE_START_CONCEPTION)
list_colors = ['goldenrod']
plot_conception_chart(list_data, list_colors)

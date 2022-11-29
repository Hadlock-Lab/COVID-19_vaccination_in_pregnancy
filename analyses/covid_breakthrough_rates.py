# Author: Samantha Piekos
# Date: 11/10/22


# load environment
dbutils.library.installPyPI("rpy2")

from datetime import date
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import statistics
import datetime
import rpy2
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from numpy.random import seed
from statsmodels.stats.proportion import proportion_confint
seed(31415)


# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.cohort_covid_pregnancy_functions
import COVID19_vaccination_in_pregnancy.utilities.calculate_fetal_growth_percentile


# define functions
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


# load maternity cohorts
df_temp = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated AS mrna WHERE mrna.ob_delivery_delivery_date <= '2021-09-22'")
print('# Number of vaccinated pregnant women delivered before 9-22-21: ' + str(df_temp.count()))
# Number of vaccinated pregnant women delivered before 9-22-21: 6529

df_cohort_maternity_vaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated")
print('# Number of vaccinated pregnant women delivered: ' + str(df_cohort_maternity_vaccinated.count()))

df_cohort_maternity_jj = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_jj")
print('# Number of vaccinated (J&J) pregnant women who have delivered: ' + str(df_cohort_maternity_jj.count()))

df_cohort_maternity_vaccinated_mrna = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_mrna")
print('# Number of vaccinated (mrna) pregnant women who have delivered: ' + str(df_cohort_maternity_vaccinated_mrna.count()))

df_cohort_maternity_moderna = df_cohort_maternity_vaccinated_mrna.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of vaccinated (Moderna) pregnant women who have delivered: ' + str(df_cohort_maternity_moderna.count()))

df_cohort_maternity_pfizer  = df_cohort_maternity_vaccinated_mrna.filter(col("first_immunization_name").like("%PFIZER%")) 
print('# Number of vaccinated (Pfizer) pregnant women who have delivered: ' + str(df_cohort_maternity_pfizer.count()))

df_cohort_mom_vaccinated_but_not_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted AS mrna_2_shots WHERE mrna_2_shots.ob_delivery_delivery_date >= date_add(mrna_2_shots.full_vaccination_date, 168) AND '2021-09-22' < mrna_2_shots.ob_delivery_delivery_date")
print('# Number of fully vaccinated pregnant patients that delivered 6+ months after obtaining full vaccination status, but were not boosted by time of delivery: ' + str(df_cohort_mom_vaccinated_but_not_boosted.count()))

df_cohort_mom_vaccinated_but_not_boosted_matched = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_but_not_boosted_matched_control")
print('# Number of vaccinated, but not boosted matched pregnant women: ' + str(df_cohort_mom_vaccinated_but_not_boosted_matched.count()))

df_cohort_mom_vaccianted_bivalent_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_bivalent_boosted")
print('# Number of pregnant people that were vaccinated and bivalent boosted, but not boosted: ' + str(df_cohort_mom_vaccianted_bivalent_boosted.count()))

df_cohort_mom_vaccianted_boosted_bivalent_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted")
print('# Number of pregnant people that were vaccinated, boosted, and bivalent boosted: ' + str(df_cohort_mom_vaccianted_boosted_bivalent_boosted.count()))

df_cohort_mom_mrna_boosted = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted AS boosted WHERE boosted.ob_delivery_delivery_date >= date_add(boosted.full_vaccination_date, 168) AND '2021-09-22' < boosted.ob_delivery_delivery_date")
print('# Number of patients that were boosted 6+ months after acheiving full vaccination status and prior to delivery: ' + str(df_cohort_mom_mrna_boosted.count()))

df_cohort_mom_mrna_1_shot = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_only")
print('# Number of people vaccinated with only 1 mRNA shot prior to delivery: ' + str(df_cohort_mom_mrna_1_shot.count()))

df_cohort_maternity_unvaccinated_covid_immunity = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_covid_immunity")
print('# Number of women with covid-induced immunity prior to a completed singleton pregnancy: ' + str(df_cohort_maternity_unvaccinated_covid_immunity.count()))

df_cohort_maternity_unvaccinated = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated AS um WHERE '2021-01-26' <= um.ob_delivery_delivery_date")
print('# Number of unvaccinated pregnant women who have delivered after 1-25-21: ' + str(df_cohort_maternity_unvaccinated.count()))

df_cohort_maternity_unvaccinated_matched = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_matched_control")
print('# Number of unvaccinated matched pregnant women: ' + str(df_cohort_maternity_unvaccinated_matched.count()))


# load covid maternity cohorts
df_breakthrough_jj = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_jj")
print('# Number of women with breakthrough covid infection (J&J) during pregnancy who has delivered: ' + str(df_breakthrough_jj.count()))

df_breakthrough_mrna = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_covid_breakthrough_mrna")
print('# Number of women with breakthrough covid infection during pregnancy who has delivered: ' + str(df_breakthrough_mrna.count()))

df_breakthrough_moderna = df_breakthrough_mrna.filter(col("first_immunization_name").like("%MODERNA%"))
print('# Number of women with breakthrough covid infection (Moderna) during pregnancy who has delivered: ' + str(df_breakthrough_moderna.count()))

df_breakthrough_pfizer = df_breakthrough_mrna.filter(col("first_immunization_name").like("%PFIZER%"))
print('# Number of women with breakthrough covid infection (Pfizer) during pregnancy who has delivered: ' + str(df_breakthrough_pfizer.count()))

df_cohort_mom_vaccinated_but_not_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_but_not_boosted_covid_breakthrough")
print('# Number of people vaccinated, but not boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count()))

df_cohort_mom_vaccinated_but_not_boosted_matched_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp3_cohort_maternity_vaccinated_but_not_boosted_matched_control_covid")
print('# Number of vaccinated, but not boosted matched pregnant women with a covid infection during pregnancy: ' + str(df_cohort_mom_vaccinated_but_not_boosted_matched_covid.count()))

df_cohort_mom_vaccinated_bivalent_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_bivalent_boosted_covid_breakthrough")
print('# Number of people vaccinated and bivalent boosted but not boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.count()))

df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_vaccinated_boosted_bivalent_boosted_covid_breakthrough")
print('# Number of people vaccinated, boosted, and bivalent boosted at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.count()))

df_cohort_mom_mrna_boosted_breakthrough = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_boosted_covid_breakthrough")
print('# Number of people boosted (mrna) at delivery with breakthrough infections during pregnancy: ' + str(df_cohort_mom_mrna_boosted_breakthrough.count()))

df_cohort_mom_mrna_1_shot_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_mrna_1_shot_covid_infection")
print('# Number of people fully vaccinated with only 1 mRNA shot before contracting covid during pregnancy: ' + str(df_cohort_mom_mrna_1_shot_covid.count()))

df_covid_immunity_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_unvaccinated_covid_immunity")
print('# Number of women with covid-induced immunity prior to pregnancy that got a subsequent COIVD-19 infection while unvaccinated during pregnancy who has delivered: ' + str(df_covid_immunity_covid.count()))

df_unvaccinated_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_covid_maternity_covid_unvaccinated AS ucm WHERE ucm.ob_delivery_delivery_date >= '2021-01-26'")
print('# Number of pregnant women unvaccinated at time of COIVD-19 infection during pregnancy who have delivered after 1-25-21: ' + str(df_unvaccinated_covid.count()))

df_unvaccinated_matched_covid = spark.sql("SELECT * FROM rdp_phi_sandbox.snp2_cohort_maternity_unvaccinated_matched_control_covid")
print('# Number of pregnancies in matched control unvaccinated cohort during which a covid infection occured: ' + str(df_unvaccinated_matched_covid.count()))


# evaluate maternal covid rates
dict_unvaccinated_covid = {'Covid': df_unvaccinated_covid.count(), 'No Covid': (df_cohort_maternity_unvaccinated.count()-df_unvaccinated_covid.count())}
conversion = dict_unvaccinated_covid['No Covid']/sum(dict_unvaccinated_covid.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_covid['Covid'], dict_unvaccinated_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_covid['Covid']/(dict_unvaccinated_covid['No Covid'] + dict_unvaccinated_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))
print('\n')

dict_induced_immunity = {'Covid': df_covid_immunity_covid.count(), 'No Covid': (df_cohort_maternity_unvaccinated_covid_immunity.count()-df_covid_immunity_covid.count())}
conversion = dict_induced_immunity['No Covid']/sum(dict_induced_immunity.values())
ci_low, ci_up = proportion_confint(dict_induced_immunity['Covid'], dict_induced_immunity['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_induced_immunity['Covid']/(dict_induced_immunity['No Covid'] + dict_induced_immunity['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_induced_immunity, dict_unvaccinated_covid)


dict_one_mrna_covid = {'Covid': df_cohort_mom_mrna_1_shot_covid.count(), 'No Covid': (df_cohort_mom_mrna_1_shot.count()-df_cohort_mom_mrna_1_shot_covid.count())}
conversion = dict_one_mrna_covid['No Covid']/sum(dict_one_mrna_covid.values())
ci_low, ci_up = proportion_confint(dict_one_mrna_covid['Covid'], dict_one_mrna_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_one_mrna_covid['Covid']/(dict_one_mrna_covid['No Covid'] + dict_one_mrna_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_one_mrna_covid, dict_unvaccinated_covid)


dict_mrna_vaccinated_covid = {'Covid': df_breakthrough_mrna.count(), 'No Covid': (df_cohort_maternity_vaccinated_mrna.count()-df_breakthrough_mrna.count())}
conversion = dict_mrna_vaccinated_covid['No Covid']/sum(dict_mrna_vaccinated_covid.values())
ci_low, ci_up = proportion_confint(dict_mrna_vaccinated_covid['Covid'], dict_mrna_vaccinated_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_mrna_vaccinated_covid['Covid']/(dict_mrna_vaccinated_covid['No Covid'] + dict_mrna_vaccinated_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_mrna_vaccinated_covid, dict_unvaccinated_covid)


dict_unvaccinated_matched_covid = {'Covid': df_unvaccinated_matched_covid.count(), 'No Covid': (df_cohort_maternity_unvaccinated_matched.count()-df_unvaccinated_matched_covid.count())}
conversion = dict_unvaccinated_matched_covid['No Covid']/sum(dict_unvaccinated_matched_covid.values())
ci_low, ci_up = proportion_confint(dict_unvaccinated_matched_covid['Covid'], dict_unvaccinated_matched_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_unvaccinated_matched_covid['Covid']/(dict_unvaccinated_matched_covid['No Covid'] + dict_unvaccinated_matched_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_mrna_vaccinated_covid, dict_unvaccinated_matched_covid)


dict_jj_vaccinated_covid = {'Covid': df_breakthrough_jj.count(), 'No Covid': (df_cohort_maternity_jj.count()-df_breakthrough_jj.count())}
conversion = dict_jj_vaccinated_covid['No Covid']/sum(dict_jj_vaccinated_covid.values())
ci_low, ci_up = proportion_confint(dict_mrna_vaccinated_covid['Covid'], dict_mrna_vaccinated_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_mrna_vaccinated_covid['Covid']/(dict_mrna_vaccinated_covid['No Covid'] + dict_mrna_vaccinated_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_jj_vaccinated_covid, dict_unvaccinated_covid)


dict_moderna_vaccinated_covid = {'Covid': df_breakthrough_moderna.count(), 'No Covid': (df_cohort_maternity_moderna.count()-df_breakthrough_moderna.count())}
conversion = dict_moderna_vaccinated_covid['No Covid']/sum(dict_moderna_vaccinated_covid.values())
ci_low, ci_up = proportion_confint(dict_moderna_vaccinated_covid['Covid'], dict_moderna_vaccinated_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_moderna_vaccinated_covid['Covid']/(dict_moderna_vaccinated_covid['No Covid'] + dict_moderna_vaccinated_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_moderna_vaccinated_covid, dict_unvaccinated_covid)


dict_pfizer_vaccinated_covid = {'Covid': df_breakthrough_pfizer.count(), 'No Covid': (df_cohort_maternity_pfizer.count()-df_breakthrough_pfizer.count())}
conversion = dict_pfizer_vaccinated_covid['No Covid']/sum(dict_pfizer_vaccinated_covid.values())
ci_low, ci_up = proportion_confint(dict_pfizer_vaccinated_covid['Covid'], dict_pfizer_vaccinated_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_pfizer_vaccinated_covid['Covid']/(dict_pfizer_vaccinated_covid['No Covid'] + dict_pfizer_vaccinated_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_pfizer_vaccinated_covid, dict_unvaccinated_covid)
calc_fishers_exact_test_2x2(dict_moderna_vaccinated_covid, dict_pfizer_vaccinated_covid)


dict_vaccinated_but_not_boosted_covid = {'Covid': df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count(), 'No Covid': (df_cohort_mom_vaccinated_but_not_boosted.count()-df_cohort_mom_vaccinated_but_not_boosted_breakthrough.count())}
conversion = dict_vaccinated_but_not_boosted_covid['No Covid']/sum(dict_vaccinated_but_not_boosted_covid.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_covid['Covid'], dict_vaccinated_but_not_boosted_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_covid['Covid']/(dict_vaccinated_but_not_boosted_covid['No Covid'] + dict_vaccinated_but_not_boosted_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))


dict_boosted_covid = {'Covid': df_cohort_mom_mrna_boosted_breakthrough.count(), 'No Covid': (df_cohort_mom_mrna_boosted.count()-df_cohort_mom_mrna_boosted_breakthrough.count())}
conversion = dict_boosted_covid['No Covid']/sum(dict_boosted_covid.values())
ci_low, ci_up = proportion_confint(dict_boosted_covid['Covid'], dict_boosted_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_boosted_covid['Covid']/(dict_boosted_covid['No Covid'] + dict_boosted_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_boosted_covid, dict_vaccinated_but_not_boosted_covid)


dict_vaccinated_but_not_boosted_matched_covid = {'Covid': df_cohort_mom_vaccinated_but_not_boosted_matched_covid.count(), 'No Covid': (df_cohort_mom_vaccinated_but_not_boosted_matched.count()-df_cohort_mom_vaccinated_but_not_boosted_matched_covid.count())}
conversion = dict_vaccinated_but_not_boosted_matched_covid['No Covid']/sum(dict_vaccinated_but_not_boosted_matched_covid.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_but_not_boosted_matched_covid['Covid'], dict_vaccinated_but_not_boosted_matched_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_but_not_boosted_matched_covid['Covid']/(dict_vaccinated_but_not_boosted_matched_covid['No Covid'] + dict_vaccinated_but_not_boosted_matched_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_boosted_covid, dict_vaccinated_but_not_boosted_matched_covid)


dict_vaccinated_bivalent_boosted_covid = {'Covid':df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.count(), 'No Covid': (df_cohort_mom_vaccianted_bivalent_boosted.count()-df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.count())}
conversion = dict_vaccinated_bivalent_boosted_covid['No Covid']/sum(dict_vaccinated_bivalent_boosted_covid.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_bivalent_boosted_covid['Covid'], dict_vaccinated_bivalent_boosted_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_bivalent_boosted_covid['Covid']/(dict_vaccinated_bivalent_boosted_covid['No Covid'] + dict_vaccinated_bivalent_boosted_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_vaccinated_bivalent_boosted_covid, dict_vaccinated_but_not_boosted_covid)
calc_fishers_exact_test_2x2(dict_vaccinated_bivalent_boosted_covid, dict_vaccinated_but_not_boosted_matched_covid)
calc_fishers_exact_test_2x2(dict_boosted_covid, dict_vaccinated_bivalent_boosted_covid)


dict_vaccinated_boosted_bivalent_boosted_covid = {'Covid': df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.count(), 'No Covid': (df_cohort_mom_vaccianted_boosted_bivalent_boosted.count()-df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.count())}
conversion = dict_vaccinated_boosted_bivalent_boosted_covid['No Covid']/sum(dict_vaccinated_boosted_bivalent_boosted_covid.values())
ci_low, ci_up = proportion_confint(dict_vaccinated_boosted_bivalent_boosted_covid['Covid'], dict_vaccinated_boosted_bivalent_boosted_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_boosted_bivalent_boosted_covid['Covid']/(dict_vaccinated_boosted_bivalent_boosted_covid['No Covid'] + dict_vaccinated_boosted_bivalent_boosted_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_vaccinated_boosted_bivalent_boosted_covid, dict_boosted_covid)
calc_fishers_exact_test_2x2(dict_vaccinated_boosted_bivalent_boosted_covid, dict_vaccinated_bivalent_boosted_covid)


count_bivalent_boosted = df_cohort_mom_vaccianted_boosted_bivalent_boosted.count() + df_cohort_mom_vaccianted_bivalent_boosted.count()
count_bivalent_boosted_breakthrough = df_cohort_mom_vaccinated_boosted_bivalent_boosted_breakthrough.count() + df_cohort_mom_vaccinated_bivalent_boosted_breakthrough.count()
dict_vaccinated_bivalent_boosted_covid = {'Covid': count_bivalent_boosted_breakthrough, 'No Covid': (count_bivalent_boosted - count_bivalent_boosted_breakthrough)}
conversion = dict_vaccinated_boosted_bivalent_boosted_covid['No Covid']/count_bivalent_boosted
ci_low, ci_up = proportion_confint(dict_vaccinated_bivalent_boosted_covid['Covid'], dict_vaccinated_bivalent_boosted_covid['No Covid'], alpha=0.05, method='wilson')
print('Percentage: ' + str(round(dict_vaccinated_bivalent_boosted_covid['Covid']/(dict_vaccinated_bivalent_boosted_covid['No Covid'] + dict_vaccinated_bivalent_boosted_covid['Covid']), 4)))
print('95% CI: ' + str(round(ci_low*conversion, 4)), str(round(ci_up*conversion, 4)))

calc_fishers_exact_test_2x2(dict_vaccinated_bivalent_boosted_covid, dict_vaccinated_but_not_boosted_covid)

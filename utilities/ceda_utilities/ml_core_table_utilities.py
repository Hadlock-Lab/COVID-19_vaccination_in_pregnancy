# Get diagnoses by encounter
# Define function for getting diagnoses by encounter
def get_diagnoses_by_encounter(encounter_table_name, cc_dictionary_file, cc_label_list):
  
  # Validate diagnosis clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  
  # Load encounter table
  encounter_df = spark.sql("SELECT * FROM rdp_phi_sandbox.db.{}".format(encounter_table_name))
  
  # Load condition label / diagnosis ID dictionary
  condition_diagnosis_id_dictionary_df = spark.read.json(cc_dictionary_file)
  condition_diagnosis_id_dictionary = condition_diagnosis_id_dictionary_df.rdd.collect()[0].asDict()

  # Get the full diagnosis_id list for selected conditions (for filtering final results)
  all_diagnosis_id_list = [id_ for label, id_list in condition_diagnosis_id_dictionary.items() for id_ in id_list if (label in cc_label_list)]
  all_diagnosis_id_list = list(set(all_diagnosis_id_list))

  # Convenience functions for creation of new columns
  def get_encounter_start(): return F.coalesce(F.col('admission_datetime').cast(DateType()), 'contact_date').alias('encounter_start')
  def get_encounter_end(): return F.coalesce(F.col('discharge_datetime').cast(DateType()), F.col('admission_datetime').cast(DateType()), 'contact_date').alias('encounter_end')
  def get_cc_col(cc_label): return F.when(F.col('exploded_diagnosis_id').isin(condition_diagnosis_id_dictionary[cc_label]), 1).otherwise(0).alias(cc_label)
  # Specify selection / creation columns
  enc_select_col_names = [
    'patient_id', 'encounter_id', 'contact_date', 'admission_datetime', 'discharge_datetime',
    'encounter_start', 'encounter_end', 'diagnosis_id', 'diagnosis_name']
  enc_new_cols = {'encounter_start': get_encounter_start(), 'encounter_end': get_encounter_end()}
  enc_select_cols = [c if not(c in enc_new_cols.keys()) else enc_new_cols[c] for c in enc_select_col_names]

  # Specify clinical concept 1/0 columns and final aggregated columns
  w = Window.partitionBy('patient_id', 'encounter_id')
  cc_cols_1_0 = [get_cc_col(c) for c in cc_label_list]
  cc_cols_agg = [F.max(c).over(w).alias(c) for c in cc_label_list]

  # Columns to drop from final
  drop_column_names = ['contact_date', 'admission_datetime', 'discharge_datetime', 'diagnosis_id']

  # Select and create desired encounter table columns
  encounter_diag_df = encounter_df \
    .select(*enc_select_cols, F.explode('diagnosis_id').alias('exploded_diagnosis_id')) \
    .where(F.col('exploded_diagnosis_id').isin(all_diagnosis_id_list)) \
    .select(*enc_select_col_names, *cc_cols_1_0) \
    .select(*enc_select_col_names, *cc_cols_agg) \
    .drop(*drop_column_names).dropDuplicates()
  
  return encounter_diag_df


# Get diagnoses by patient
# Define function for getting diagnoses by patient
def get_diagnoses_by_patient(enc_diag_ml_table, cc_label_list):
  
  # Validate diagnosis clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  
  # Load diagnoses-by-encounter ML table and confirm it contains required clinical concept columns
  enc_diag_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(enc_diag_ml_table))
  assert(all([(label in enc_diag_df.columns) for label in cc_label_list]))
  
  # Convenience functions for aggregation of encounter information
  w1 = Window.partitionBy('patient_id').orderBy('encounter_start', 'encounter_end') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  def get_first_enc_date(): return F.first('encounter_start').over(w1).alias('first_encounter_start')
  def get_last_enc_date(): return F.last('encounter_end').over(w1).alias('last_encounter_end')
  def get_encounter_ids(): return F.collect_list('encounter_id').over(w1).alias('encounter_ids')
  def get_diagnosis_names(): return F.collect_list('diagnosis_name').over(w1).alias('diagnosis_names')
  
  # Specify selection / creation columns
  pat_select_col_names = ['patient_id', 'encounter_ids', 'first_encounter_start', 'last_encounter_end', 'diagnosis_names']
  pat_new_cols = {
    'encounter_ids': get_encounter_ids(), 'first_encounter_start': get_first_enc_date(),
    'last_encounter_end': get_last_enc_date(), 'diagnosis_names': get_diagnosis_names()}
  pat_select_cols = [c if not(c in pat_new_cols.keys()) else pat_new_cols[c] for c in pat_select_col_names]
  
  # Specify initial first and last date columns for clinical concepts
  first_date_col_names = ['{}_first'.format(c) for c in cc_label_list]
  last_date_col_names = ['{}_last'.format(c) for c in cc_label_list]
  first_date_cols = [F.when(F.col(c) == 1, F.col('encounter_start')).otherwise(None).alias('{}_first'.format(c)) for c in cc_label_list]
  last_date_cols = [F.when(F.col(c) == 1, F.col('encounter_end')).otherwise(None).alias('{}_last'.format(c)) for c in cc_label_list]
  
  # Add first and last date columns
  enc_diag_df = enc_diag_df \
    .select(*pat_select_cols, *cc_label_list, *first_date_cols, *last_date_cols)
  
  # Specify final clinical concept aggregation columns
  w2 = Window.partitionBy('patient_id')
  cc_cols_1_0 = [F.max(c).over(w2).alias(c) for c in cc_label_list]
  first_date_cols = [F.min(c).over(w2).alias(c) for c in first_date_col_names]
  last_date_cols = [F.max(c).over(w2).alias(c) for c in last_date_col_names]
  
  # Add final clinical concept columns
  final_select_col_names = [item_ for tuple_ in zip(cc_label_list, first_date_col_names, last_date_col_names) for item_ in tuple_]
  pat_diag_df = enc_diag_df \
    .select(*pat_select_col_names, *cc_cols_1_0, *first_date_cols, *last_date_cols) \
    .select(*pat_select_col_names, *final_select_col_names) \
    .dropDuplicates()
  
  return pat_diag_df


# Get medications by encounter
# Define function for getting medications by encounter
def get_med_orders_by_encounter(med_order_table_name, cc_dictionary_file, cc_label_list):
  
  # Validate medication clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  
  # Load medication order table
  med_order_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(med_order_table_name))
  
  # Load medication label / medication ID dictionary
  medication_medication_id_dictionary_df = spark.read.json(cc_dictionary_file)
  medication_medication_id_dictionary = medication_medication_id_dictionary_df.rdd.collect()[0].asDict()

  # Get the full medication_id list for selected medications (for filtering final results)
  all_medication_id_list = [id_ for label, id_list in medication_medication_id_dictionary.items() for id_ in id_list if (label in cc_label_list)]
  all_medication_id_list = list(set(all_medication_id_list))

  # Convenience functions for creation of new columns
  w = Window.partitionBy('patient_id', 'encounter_id')
  def get_first_med_order_date(): return F.min('ordering_datetime').over(w).cast(DateType()).alias('first_med_order_date')
  def get_last_med_order_date(): return F.max('ordering_datetime').over(w).cast(DateType()).alias('last_med_order_date')
  def collect_med_descriptions(): return F.collect_set('order_description').over(w).alias('medication_descriptions')
  def get_cc_col(cc_label): return F.when(F.col('medication_id').isin(medication_medication_id_dictionary[cc_label]), 1).otherwise(0).alias(cc_label)

  # Specify selection / creation columns
  med_select_col_names = [
    'patient_id', 'encounter_id', 'ordering_datetime', 'medication_id', 'order_description']
  med_new_cols = {
    'first_med_order_date': get_first_med_order_date(),
    'last_med_order_date': get_last_med_order_date(),
    'medication_descriptions': collect_med_descriptions()}
  med_additional_col_names = list(med_new_cols.keys())

  # Specify clinical concept 1/0 columns and final aggregated columns
  cc_cols_1_0 = [get_cc_col(c) for c in cc_label_list]
  cc_cols_agg = [F.max(c).over(w).alias(c) for c in cc_label_list]

  # Columns to drop from final
  drop_column_names = ['ordering_datetime', 'medication_id', 'order_description']

  # Select and create desired medication table columns
  encounter_meds_df = med_order_df \
    .select(*med_select_col_names).dropDuplicates() \
    .where(F.col('medication_id').isin(all_medication_id_list)) \
    .select(*med_select_col_names, *[med_new_cols[c] for c in med_additional_col_names], *cc_cols_1_0) \
    .select(*med_select_col_names, *med_additional_col_names, *cc_cols_agg) \
    .drop(*drop_column_names).dropDuplicates()
  
  return encounter_meds_df


# Get medications by patient
# Define function for getting medications by patient
def get_med_orders_by_patient(enc_meds_ml_table, cc_label_list):
  
  # Validate medication clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  
  # Load medications-by-encounter ML table and confirm it contains required clinical concept columns
  enc_meds_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(enc_meds_ml_table))
  assert(all([(label in enc_meds_df.columns) for label in cc_label_list]))
  
  # Convenience functions for aggregation of encounter information
  w1 = Window.partitionBy('patient_id').orderBy('first_med_order_date', 'last_med_order_date') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  def get_first_order_date(): return F.first('first_med_order_date').over(w1).alias('first_med_order_date')
  def get_last_order_date(): return F.last('last_med_order_date').over(w1).alias('last_med_order_date')
  def get_encounter_ids(): return F.collect_list('encounter_id').over(w1).alias('encounter_ids')
  def get_medication_descriptions(): return F.collect_list('medication_descriptions').over(w1).alias('medication_descriptions')

  # Specify selection / creation columns
  pat_select_col_names = ['patient_id', 'encounter_ids', 'first_med_order_date', 'last_med_order_date', 'medication_descriptions']
  pat_new_cols = {
    'encounter_ids': get_encounter_ids(), 'first_med_order_date': get_first_order_date(),
    'last_med_order_date': get_last_order_date(), 'medication_descriptions': get_medication_descriptions()}
  pat_select_cols = [c if not(c in pat_new_cols.keys()) else pat_new_cols[c] for c in pat_select_col_names]
  
  # Specify initial first and last date columns for clinical concepts
  first_date_col_names = ['{}_first'.format(c) for c in cc_label_list]
  last_date_col_names = ['{}_last'.format(c) for c in cc_label_list]
  first_date_cols = [F.when(F.col(c) == 1, F.col('first_med_order_date')).otherwise(None).alias('{}_first'.format(c)) for c in cc_label_list]
  last_date_cols = [F.when(F.col(c) == 1, F.col('last_med_order_date')).otherwise(None).alias('{}_last'.format(c)) for c in cc_label_list]
  
  # Add first and last date columns
  enc_meds_df = enc_meds_df \
    .select(*pat_select_cols, *cc_label_list, *first_date_cols, *last_date_cols)
  
  # Specify final clinical concept aggregation columns
  w2 = Window.partitionBy('patient_id')
  cc_cols_1_0 = [F.max(c).over(w2).alias(c) for c in cc_label_list]
  first_date_cols = [F.min(c).over(w2).alias(c) for c in first_date_col_names]
  last_date_cols = [F.max(c).over(w2).alias(c) for c in last_date_col_names]
  
  # Add final clinical concept columns
  final_select_col_names = [item_ for tuple_ in zip(cc_label_list, first_date_col_names, last_date_col_names) for item_ in tuple_]
  pat_meds_df = enc_meds_df \
    .select(*pat_select_col_names, *cc_cols_1_0, *first_date_cols, *last_date_cols) \
    .select(*pat_select_col_names, *final_select_col_names) \
    .dropDuplicates()
  
  return pat_meds_df


# Get labs by encounter
# Define function for getting labs by encounter
def get_labs_by_encounter(proc_order_table_name, cc_dictionary_file, cc_label_list, date_bounds=None):
  
  # Validate lab clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Get separate lists for 'high/low' labs, 'presence' labs, and 'pos/neg' labs
  cc_label_list_pos_neg = [label for label in cc_label_list if (label in get_pos_neg_label_list())]
  cc_label_list_high_low = [label for label in cc_label_list if (label in get_high_low_label_list())]
  cc_label_list_presence = [label for label in cc_label_list if (label in get_presence_label_list())]
  
  # Load procedure order table
  proc_order_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(proc_order_table_name))
  proc_order_all_col_names = proc_order_df.columns
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting problem list records to between '{0}' and '{1}'...".format(*date_bounds))
    proc_order_df = proc_order_df \
      .where(F.col('ordering_datetime').between(*valid_date_bounds))
  
  # Load lab label / lab ID dictionary
  lab_lab_id_dictionary_df = spark.read.json(cc_dictionary_file)
  lab_lab_id_dictionary = lab_lab_id_dictionary_df.rdd.collect()[0].asDict()
  
  # Get the full lab_id list for selected labs (for filtering final results)
  all_lab_id_list = [id_ for label, id_list in lab_lab_id_dictionary.items() for id_ in id_list if (label in cc_label_list)]
  all_lab_id_list = list(set(all_lab_id_list))

  # Convenience functions for creation of new columns
  w = Window.partitionBy('patient_id', 'encounter_id')
  def flagged_as_high(): return F.col('flagged_as').isin(['High', 'High Panic'])
  def flagged_as_low(): return F.col('flagged_as').isin(['Low', 'Low Panic'])
  def flagged_as_abnormal(): return F.col('flagged_as').isin(['Abnormal', 'High'])
  def lab_id_in_cc_category(label): return F.col('lab_id').isin(lab_lab_id_dictionary[label])
  def get_first_lab_order_date(): return F.min('ordering_datetime').over(w).cast(DateType()).alias('first_lab_order_date')
  def get_last_lab_order_date(): return F.max('ordering_datetime').over(w).cast(DateType()).alias('last_lab_order_date')
  def collect_lab_descriptions(): return F.collect_set('lab_id').over(w).alias('lab_descriptions')
  def get_high_flag(c): return F.when(lab_id_in_cc_category(c) & flagged_as_high(), 1).otherwise(0).alias('{}_high'.format(c))
  def get_low_flag(c): return F.when(lab_id_in_cc_category(c) & flagged_as_low(), 1).otherwise(0).alias('{}_low'.format(c))
  def get_presence_flag(c): return F.when(lab_id_in_cc_category(c) & flagged_as_abnormal(), 1).otherwise(0).alias('{}_abnormal'.format(c))
  def get_query_string_list(c):
    return ['({})'.format(get_field_search_query_string(f, slist, False)) for f, slist in cc_lab_search_strings[c].items()]
  def get_field_search_match(c): return F.when(F.expr(' AND '.join(get_query_string_list(c))), True).otherwise(False).alias('{}_match'.format(c))
  def get_positive_result_flag(c):
    return F.when(F.col('{}_match'.format(c)) & (classify_pcr_naat_outcome_udf('result_value', F.lit(c)) == 'positive'), 1).otherwise(0).alias('{}_pos'.format(c))
  def get_negative_result_flag(c):
    return F.when(F.col('{}_match'.format(c)) & (classify_pcr_naat_outcome_udf('result_value', F.lit(c)) == 'negative'), 1).otherwise(0).alias('{}_neg'.format(c))
  
  # Specify columns for clinical concepts
  high_low_col_names = [item_ for label in cc_label_list_high_low for item_ in ('{}_high'.format(label), '{}_low'.format(label))]
  high_cols = [get_high_flag(label) for label in cc_label_list_high_low]
  low_cols = [get_low_flag(label) for label in cc_label_list_high_low]
  presence_col_names = ['{}_abnormal'.format(label) for label in cc_label_list_presence]
  presence_cols = [get_presence_flag(label) for label in cc_label_list_presence]
  pos_neg_col_names = [item_ for label in cc_label_list_pos_neg for item_ in ('{}_pos'.format(label), '{}_neg'.format(label))]
  pos_cols = [get_positive_result_flag(label) for label in cc_label_list_pos_neg]
  neg_cols = [get_negative_result_flag(label) for label in cc_label_list_pos_neg]
  
  # Specify final aggregated clinical concept columns
  all_cc_col_names = high_low_col_names + presence_col_names + pos_neg_col_names
  final_cc_cols = [F.max(label).over(w).alias(label) for label in all_cc_col_names]
  
  # Specify selection / creation columns
  lab_select_col_names = [
    'patient_id', 'encounter_id', 'ordering_datetime', 'lab_id', 'flagged_as']
  lab_new_cols = {
    'first_lab_order_date': get_first_lab_order_date(),
    'last_lab_order_date': get_last_lab_order_date(),
    'lab_descriptions': collect_lab_descriptions()}
  lab_additional_col_names = list(lab_new_cols.keys())
  lab_additional_cols = [lab_new_cols[c] for c in lab_additional_col_names]

  # Columns to drop from final
  drop_column_names = ['ordering_datetime', 'lab_id', 'flagged_as']

  # Select and create desired lab table columns
  encounter_labs_df = proc_order_df \
    .select(*proc_order_all_col_names, *[get_field_search_match(c) for c in cc_label_list_pos_neg]) \
    .withColumn('any_pos_neg_match', F.expr(' OR '.join(['({}_match = TRUE)'.format(c) for c in cc_label_list_pos_neg]))) \
    .where(~F.col('lab_id').isin(['1000', '2000', '4000'])) \
    .where(F.col('any_pos_neg_match') | F.col('lab_id').isin(all_lab_id_list)) \
    .select(*lab_select_col_names, *lab_additional_cols, *high_cols, *low_cols, *presence_cols, *pos_cols, *neg_cols) \
    .select(*lab_select_col_names, *lab_additional_col_names, *final_cc_cols) \
    .drop(*drop_column_names).dropDuplicates()
  
  return encounter_labs_df


# Get labs by patient
# Define function for getting labs by patient
def get_labs_by_patient(enc_labs_ml_table, cc_label_list):
  # Validate lab clinical concept labels
  assert(cc_labels_are_valid(cc_label_list))
  # Get separate lists for 'high/low', 'presence', and 'pos/neg' labs
  cc_label_list_pos_neg = [label for label in cc_label_list if (label in get_pos_neg_label_list())]
  cc_label_list_high_low = [label for label in cc_label_list if (label in get_high_low_label_list())]
  cc_label_list_presence = [label for label in cc_label_list if (label in get_presence_label_list())]
  
  # Get all column names
  pos_neg_col_names = [item_ for label in cc_label_list_pos_neg for item_ in ('{}_pos'.format(label), '{}_neg'.format(label))]
  high_low_col_names = [item_ for label in cc_label_list_high_low for item_ in ('{}_high'.format(label), '{}_low'.format(label))]
  presence_col_names = ['{}_abnormal'.format(label) for label in cc_label_list_presence]
  all_cc_col_names = high_low_col_names + presence_col_names + pos_neg_col_names
  
  # Load labs-by-encounter ML table and confirm it contains required clinical concept columns
  enc_labs_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(enc_labs_ml_table))
  assert(all([(label in enc_labs_df.columns) for label in all_cc_col_names]))
  
  # Convenience functions for aggregation of encounter information
  w1 = Window.partitionBy('patient_id').orderBy('first_lab_order_date', 'last_lab_order_date') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  def get_first_order_date(): return F.first('first_lab_order_date').over(w1).alias('first_lab_order_date')
  def get_last_order_date(): return F.last('last_lab_order_date').over(w1).alias('last_lab_order_date')
  def get_encounter_ids(): return F.collect_list('encounter_id').over(w1).alias('encounter_ids')
  
  # Specify selection / creation columns
  pat_select_col_names = ['patient_id', 'encounter_ids', 'first_lab_order_date', 'last_lab_order_date']
  pat_new_cols = {
    'encounter_ids': get_encounter_ids(), 'first_lab_order_date': get_first_order_date(),
    'last_lab_order_date': get_last_order_date()}
  pat_select_cols = [c if not(c in pat_new_cols.keys()) else pat_new_cols[c] for c in pat_select_col_names]
  
  # Specify initial first and last date columns for clinical concepts
  first_date_col_names = ['{}_first'.format(c) for c in all_cc_col_names]
  last_date_col_names = ['{}_last'.format(c) for c in all_cc_col_names]
  first_date_cols = [F.when(F.col(c) == 1, F.col('first_lab_order_date')).otherwise(None).alias('{}_first'.format(c)) for c in all_cc_col_names]
  last_date_cols = [F.when(F.col(c) == 1, F.col('last_lab_order_date')).otherwise(None).alias('{}_last'.format(c)) for c in all_cc_col_names]
  
  # Add first and last date columns
  enc_labs_df = enc_labs_df \
    .select(*pat_select_cols, *all_cc_col_names, *first_date_cols, *last_date_cols)
  
  # Specify final clinical concept aggregation columns
  w2 = Window.partitionBy('patient_id')
  cc_cols_1_0 = [F.max(c).over(w2).alias(c) for c in all_cc_col_names]
  first_date_cols = [F.min(c).over(w2).alias(c) for c in first_date_col_names]
  last_date_cols = [F.max(c).over(w2).alias(c) for c in last_date_col_names]
  
  # Add final clinical concept columns
  final_select_col_names = [item_ for tuple_ in zip(all_cc_col_names, first_date_col_names, last_date_col_names) for item_ in tuple_]
  pat_labs_df = enc_labs_df \
    .select(*pat_select_col_names, *cc_cols_1_0, *first_date_cols, *last_date_cols) \
    .select(*pat_select_col_names, *final_select_col_names) \
    .dropDuplicates()
  
  return pat_labs_df


# Merge patient ML tables
def merge_ml_patient_tables_with_race(diag_table_name, meds_table_name, labs_table_name, enc_table_name):
  
  # Load patient diagnoses
  pat_diag_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(diag_table_name)).drop('encounter_ids')
  non_diag_cc_col_names = ['first_encounter_start', 'last_encounter_end', 'diagnosis_names']
  diag_cc_col_names = [c for c in pat_diag_df.columns if not(c in non_diag_cc_col_names + ['patient_id'])]

  # Load patient medications
  pat_meds_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(meds_table_name)).drop('encounter_ids')
  non_meds_cc_col_names = ['first_med_order_date', 'last_med_order_date', 'medication_descriptions']
  meds_cc_col_names = [c for c in pat_meds_df.columns if not(c in non_meds_cc_col_names + ['patient_id'])]

  # Load patient labs
  pat_labs_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(labs_table_name)).drop('encounter_ids')
  non_labs_cc_col_names = ['first_lab_order_date', 'last_lab_order_date']
  labs_cc_col_names = [c for c in pat_labs_df.columns if not(c in non_labs_cc_col_names + ['patient_id'])]

  # Define race column
  race_col = F.when(F.array_contains('race', 'American Indian or Alaska Native'), 'American Indian or Alaska Native') \
    .when(F.array_contains('race', 'Native Hawaiian or Other Pacific Islander'), 'Native Hawaiian or Other Pacific Islander') \
    .when(F.array_contains('race', 'Black or African American'), 'Black or African American') \
    .when(F.array_contains('race', 'Asian'), 'Asian') \
    .when(F.array_contains('race', 'White or Caucasian'), 'White or Caucasian') \
    .when(F.array_contains('race', 'White'), 'White or Caucasian') \
    .when(F.array_contains('race', 'Other'), 'Other') \
    .when(F.array_contains('race', 'Unknown'), 'Unknown') \
    .when(F.array_contains('race', 'Patient Refused'), 'Patient Refused') \
    .when(F.array_contains('race', 'Declined'), 'Declined') \
    .when(F.array_contains('race', 'Unable to Determine'), 'Unable to Determine') \
    .otherwise(F.lit(None)).alias('race')
  
  # Combine non-CC and CC column names
  all_non_cc_col_names =  \
    ['patient_id', 'birth_date', 'sex', 'ethnic_group', 'race', 'first_record', 'last_record'] + \
    non_diag_cc_col_names + non_meds_cc_col_names + non_labs_cc_col_names
  all_cc_col_names = diag_cc_col_names + meds_cc_col_names + labs_cc_col_names
  non_date_cc_col_names = [c for c in all_cc_col_names if not(any([(suffix in c) for suffix in ['_first', '_last']]))]
  date_cc_col_names = [c for c in all_cc_col_names if any([(suffix in c) for suffix in ['_first', '_last']])]
  cc_cols_fill_na = [F.when(F.col(c).isNull(), 0).otherwise(F.col(c)).alias(c) for c in non_date_cc_col_names]

  # Get union of patient IDs
  all_patient_ids_df = pat_diag_df.select('patient_id') \
    .union(pat_meds_df.select('patient_id')) \
    .union(pat_labs_df.select('patient_id')) \
    .dropDuplicates()
  print("Total unique patient count: {}".format(all_patient_ids_df.count()))

  # Get patient demographics
  patient_demographics_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(enc_table_name)) \
    .select('patient_id', 'birth_date', 'sex', 'ethnic_group', race_col).dropDuplicates()
  print("Patient demographics table count: {}".format(patient_demographics_df.count()))

  # Get demographics only for patients in the ML tables
  all_patient_info_df = all_patient_ids_df.join(patient_demographics_df, 'patient_id')

  # Merge all tables
  merged_patient_table_df = all_patient_info_df \
    .join(pat_diag_df, ['patient_id'], how='left') \
    .join(pat_meds_df, ['patient_id'], how='left') \
    .join(pat_labs_df, ['patient_id'], how='left') \
    .withColumn('first_record', F.least('first_encounter_start', 'first_med_order_date', 'first_lab_order_date')) \
    .withColumn('last_record', F.greatest('last_encounter_end', 'last_med_order_date', 'last_lab_order_date')) \
    .select(*all_non_cc_col_names, *cc_cols_fill_na, *date_cc_col_names) \
    .select(*all_non_cc_col_names, *all_cc_col_names)
  print("Patient count in merged table: {}".format(merged_patient_table_df.count()))
  
  return merged_patient_table_df


# Merge encounter ML tables
def merge_ml_encounter_tables_with_race(diag_table_name, meds_table_name, labs_table_name, enc_table_name):
  
  # Load encounter diagnoses
  enc_diag_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(diag_table_name))
  non_diag_cc_col_names = ['encounter_start', 'encounter_end', 'diagnosis_name']
  diag_cc_col_names = [c for c in enc_diag_df.columns if not(c in non_diag_cc_col_names + ['patient_id', 'encounter_id'])]

  # Load encounter medications
  enc_meds_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(meds_table_name))
  non_meds_cc_col_names = ['first_med_order_date', 'last_med_order_date', 'medication_descriptions']
  meds_cc_col_names = [c for c in enc_meds_df.columns if not(c in non_meds_cc_col_names + ['patient_id', 'encounter_id'])]

  # Load encounter labs
  enc_labs_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(labs_table_name))
  non_labs_cc_col_names = ['first_lab_order_date', 'last_lab_order_date', 'lab_descriptions']
  labs_cc_col_names = [c for c in enc_labs_df.columns if not(c in non_labs_cc_col_names + ['patient_id', 'encounter_id'])]
  
  # Define race column
  race_col = F.when(F.array_contains('race', 'American Indian or Alaska Native'), 'American Indian or Alaska Native') \
    .when(F.array_contains('race', 'Native Hawaiian or Other Pacific Islander'), 'Native Hawaiian or Other Pacific Islander') \
    .when(F.array_contains('race', 'Black or African American'), 'Black or African American') \
    .when(F.array_contains('race', 'Asian'), 'Asian') \
    .when(F.array_contains('race', 'White or Caucasian'), 'White or Caucasian') \
    .when(F.array_contains('race', 'White'), 'White or Caucasian') \
    .when(F.array_contains('race', 'Other'), 'Other') \
    .when(F.array_contains('race', 'Unknown'), 'Unknown') \
    .when(F.array_contains('race', 'Patient Refused'), 'Patient Refused') \
    .when(F.array_contains('race', 'Declined'), 'Declined') \
    .when(F.array_contains('race', 'Unable to Determine'), 'Unable to Determine') \
    .otherwise(F.lit(None)).alias('race')
  
  # Combine non-CC and CC column names
  all_non_cc_col_names =  \
    ['patient_id', 'birth_date', 'age', 'sex', 'ethnic_group', 'race', 'encounter_id', 'first_record', 'last_record'] + \
    non_diag_cc_col_names + non_meds_cc_col_names + non_labs_cc_col_names
  all_cc_col_names = diag_cc_col_names + meds_cc_col_names + labs_cc_col_names
  cc_cols_fill_na = [F.when(F.col(c).isNull(), 0).otherwise(F.col(c)).alias(c) for c in all_cc_col_names]

  # Get union of patient IDs
  all_patient_and_encounter_ids_df = enc_diag_df.select('patient_id', 'encounter_id') \
    .union(enc_meds_df.select('patient_id', 'encounter_id')) \
    .union(enc_labs_df.select('patient_id', 'encounter_id')) \
    .dropDuplicates()
  print("Total unique patient/encounter count: {}".format(all_patient_and_encounter_ids_df.count()))

  # Get patient demographics
  patient_demographics_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(enc_table_name)) \
    .select('patient_id', 'encounter_id', 'birth_date', 'sex', 'ethnic_group', race_col).dropDuplicates()
  print("Patient demographics table count: {}".format(patient_demographics_df.count()))

  # Get demographics only for patients in the ML tables
  all_patient_info_df = all_patient_and_encounter_ids_df.join(patient_demographics_df, ['patient_id', 'encounter_id'])

  # Merge all tables
  merged_encounter_table_df = all_patient_info_df \
    .join(enc_diag_df, ['patient_id', 'encounter_id'], how='left') \
    .join(enc_meds_df, ['patient_id', 'encounter_id'], how='left') \
    .join(enc_labs_df, ['patient_id', 'encounter_id'], how='left') \
    .withColumn('first_record', F.least('encounter_start', 'first_med_order_date', 'first_lab_order_date')) \
    .withColumn('last_record', F.greatest('encounter_end', 'last_med_order_date', 'last_lab_order_date')) \
    .withColumn('age', F.round(F.datediff(F.col('first_record').cast(DateType()), F.col('birth_date'))/365.25, 1)) \
    .select(*all_non_cc_col_names, *cc_cols_fill_na)
  print("Patient/encounter count in merged table: {}".format(merged_encounter_table_df.count()))
  
  return merged_encounter_table_df


# Generate ML features table by encounter
def generate_ml_features_table_with_race_by_encounter(condition_feature_defs, med_feature_defs, lab_feature_defs, ml_encounter_table):
  
  # Get full list of features referenced in feature definitions
  all_feature_definitions = condition_feature_defs
  all_feature_definitions.update(med_feature_defs)
  all_feature_definitions.update(lab_feature_defs)
  all_referenced_features = [item_ for label, def_ in all_feature_definitions.items() for item_ in def_['cc_list']]

  # Confirm that all referenced features are valid feature labels
  all_valid_feature_labels = get_all_feature_labels()
  assert(all([(label in all_valid_feature_labels) for label in all_referenced_features]))

  # Load ML encounter table and select patients >= 18 years of age
  ml_encounters_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(ml_encounter_table))

  # Specify non-feature column names
  non_feature_col_names = [
    'patient_id', 'age', 'sex', 'ethnic_group', 'race', 'encounter_id', 'first_record', 'last_record']

  # Specify demographic feature columns
  age_feature_col = F.when(F.col('age') >= 65.0, 1).otherwise(0).alias('age_gte_65')
  sex_feature_col = F.when(F.col('sex') == 'Male', 1).otherwise(0).alias('male_sex')
  ethnic_group_feature_col = \
    F.when(F.col('ethnic_group') == 'Hispanic or Latino', 1).otherwise(0).alias('hispanic')

  # Specify feature columns
  def get_derived_feature_string(feature_list):
    return ' OR '.join(["({} == 1)".format(f) for f in feature_list])
  def get_case_when_expression(feature_str, name):
    return F.when(F.expr(feature_str), 1).otherwise(0).alias(name)
  derived_feature_cols = [get_case_when_expression(get_derived_feature_string(def_['cc_list']), name) for name, def_ in all_feature_definitions.items()]

  # Generate final table
  feature_table_df = ml_encounters_df \
    .select(*non_feature_col_names, age_feature_col, sex_feature_col, ethnic_group_feature_col, *derived_feature_cols)
  
  return feature_table_df


# Generate ML features table by patient
def generate_ml_features_table_with_race_by_patient(condition_feature_defs, med_feature_defs, lab_feature_defs, ml_patient_table):
  
  # Get full list of features referenced in feature definitions
  all_feature_definitions = condition_feature_defs
  all_feature_definitions.update(med_feature_defs)
  all_feature_definitions.update(lab_feature_defs)
  all_referenced_features = [item_ for label, def_ in all_feature_definitions.items() for item_ in def_['cc_list']]

  # Confirm that all referenced features are valid feature labels
  all_valid_feature_labels = get_all_feature_labels()
  assert(all([(label in all_valid_feature_labels) for label in all_referenced_features]))

  # Load ML patient table
  ml_patients_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(ml_patient_table))

  # Specify non-feature column names
  non_feature_col_names = [
    'patient_id', 'sex', 'ethnic_group', 'race', 'first_record', 'last_record']

  # Specify demographic feature columns
  sex_feature_col = F.when(F.col('sex') == 'Male', 1).otherwise(0).alias('male_sex')
  ethnic_group_feature_col = \
    F.when(F.col('ethnic_group') == 'Hispanic or Latino', 1).otherwise(0).alias('hispanic')

  # Utility functions to specify derived feature columns
  def get_derived_feature_string(feature_list):
    return ' OR '.join(["({} == 1)".format(f) for f in feature_list])
  def get_case_when_expression(feature_str, name):
    return F.when(F.expr(feature_str), 1).otherwise(0).alias(name)
  
  # Utility functions to specify columns to get earliest and latest dates for all features
  # Include dummy Null column in 'least' and 'greatest' to guarantee >= 2 columns as arguments
  def get_derived_feature_least(feature_list, name):
    return F.least(*[F.col('{}_first'.format(f)) for f in feature_list], F.lit(None)).alias('{}_first'.format(name))
  def get_derived_feature_greatest(feature_list, name):
    return F.greatest(*[F.col('{}_last'.format(f)) for f in feature_list], F.lit(None)).alias('{}_last'.format(name))
  
  # Specify derived columns
  derived_feature_cols = [get_case_when_expression(get_derived_feature_string(def_['cc_list']), name) for name, def_ in all_feature_definitions.items()]
  derived_feature_first_cols = [get_derived_feature_least(def_['cc_list'], name) for name, def_ in all_feature_definitions.items()]
  derived_feature_last_cols = [get_derived_feature_greatest(def_['cc_list'], name) for name, def_ in all_feature_definitions.items()]
  
  # Generate list of derived columns in correct order
  cc_feature_cols = [item_ for tuple_ in zip(derived_feature_cols, derived_feature_first_cols, derived_feature_last_cols) for item_ in tuple_]

  # Generate final table
  feature_table_df = ml_patients_df \
    .select(*non_feature_col_names, sex_feature_col, ethnic_group_feature_col, *cc_feature_cols)
  
  return feature_table_df
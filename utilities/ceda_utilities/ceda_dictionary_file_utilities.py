# Define basic utilities
# Define utility function
def get_sorted_unique_list(list_):
  list_ = list(set(list_))
  list_.sort()
  return list_


# Generate mapping from condition labels to diagnosis IDs
# Generate condition label to diagnosis ID list map
def generate_condition_label_to_diagnosis_id_map(snomed_diagnosis_id_map_table, label_to_snomed_dictionary):

  # Load SNOMED / diagnosis ID map table
  table_name = snomed_diagnosis_id_map_table
  snomed_map_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name))

  # Get conditions list
  conditions_list = list(label_to_snomed_dictionary.keys()); conditions_list.sort()
  data_temp = [(c, label_to_snomed_dictionary[c]) for c in conditions_list]

  # Create clinical concept map dataframe
  columns = ['label', 'snomed_list']
  w = Window.partitionBy('label').orderBy('diagnosis_id') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  condition_list_df = spark.createDataFrame(data_temp, columns) \
    .select('label', F.explode('snomed_list').alias('snomed')) \
    .join(snomed_map_df, ['snomed'], how='left') \
    .select('label', 'snomed', F.explode('diagnosis_id').alias('diagnosis_id')).dropDuplicates() \
    .select('label', F.collect_list('diagnosis_id').over(w).alias('diagnosis_id')).dropDuplicates() \
    .orderBy('label')
  condition_list_rdd = condition_list_df.rdd.collect()

  # Generate final condition label to diagnosis IDs map
  condition_diagnosis_id_dictionary = {row.label: get_sorted_unique_list(row.diagnosis_id) for row in condition_list_rdd}
  
  return condition_diagnosis_id_dictionary


# Generate mapping from medication labels to medication IDs
# Generate medication label to medication ID list map
def generate_medication_label_to_medication_id_map(rxnorm_medication_id_map_table, label_to_rxnorm_dictionary):

  # Load RxNorm / medication ID map table
  table_name = rxnorm_medication_id_map_table
  rxnorm_map_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name))

  # Get medications list
  medications_list = list(label_to_rxnorm_dictionary.keys()); medications_list.sort()
  data_temp = [(c, label_to_rxnorm_dictionary[c]) for c in medications_list]

  # Create clinical concept map dataframe
  columns = ['label', 'rxnorm_list']
  w = Window.partitionBy('label').orderBy('medication_id') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  medication_list_df = spark.createDataFrame(data_temp, columns) \
    .select('label', F.explode('rxnorm_list').alias('rxnorm')) \
    .join(rxnorm_map_df, ['rxnorm'], how='left') \
    .select('label', 'rxnorm', F.explode('medication_id').alias('medication_id')).dropDuplicates() \
    .select('label', F.collect_list('medication_id').over(w).alias('medication_id')).dropDuplicates() \
    .orderBy('label')
  medication_list_rdd = medication_list_df.rdd.collect()

  # Generate final medication label to medication IDs map
  medication_medication_id_dictionary = {row.label: get_sorted_unique_list(row.medication_id) for row in medication_list_rdd}
  
  return medication_medication_id_dictionary


# Generate mapping from lab labels to lab IDs
# Generate lab label to lab ID list map
def generate_lab_label_to_lab_id_map(loinc_lab_id_map_table, label_to_loinc_dictionary):

  # Load LOINC / lab ID map table
  table_name = loinc_lab_id_map_table
  loinc_map_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name))

  # Get labs list
  labs_list = list(label_to_loinc_dictionary.keys()); labs_list.sort()
  data_temp = [(c, label_to_loinc_dictionary[c]) for c in labs_list]

  # Create clinical concept map dataframe
  columns = ['label', 'loinc_list']
  w = Window.partitionBy('label').orderBy('lab_id') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  lab_list_df = spark.createDataFrame(data_temp, columns) \
    .select('label', F.explode('loinc_list').alias('loinc')) \
    .join(loinc_map_df, ['loinc'], how='left') \
    .select('label', 'loinc', F.explode('lab_id').alias('lab_id')).dropDuplicates() \
    .select('label', F.collect_list('lab_id').over(w).alias('lab_id')).dropDuplicates() \
    .orderBy('label')
  lab_list_rdd = lab_list_df.rdd.collect()

  # Generate final lab label to lab IDs map
  lab_lab_id_dictionary = {row.label: get_sorted_unique_list(row.lab_id) for row in lab_list_rdd}
  
  return lab_lab_id_dictionary
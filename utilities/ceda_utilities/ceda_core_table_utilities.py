# import functions from other notebooks
import COVID19_vaccination_in_pregnancy.utilities.ceda_utilities.ceda_tables_column_maps


# get patient table
# Generate patient table, joining related tables
def get_patient_table(db='rdp_phi', table_prefix=None):
  # Load patient table
  print("Loading patient table...")
  table_name = 'patient' if (table_prefix is None) else '{}_patient'.format(table_prefix)
  patient_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in patient_ceda_table_map.items()])

  # Load patient race table and aggregate race field
  print("Loading patient race table...")
  table_name = 'patientrace' if (table_prefix is None) else '{}_patientrace'.format(table_prefix)
  partition_by_cols = ['patient_id']
  collect_list_cols = ['race']
  w = Window.partitionBy(*partition_by_cols)
  patient_race_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in patient_race_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()

  # Join patient tables
  join_cols = ['patient_id']
  patient_all_data_df = patient_df \
    .join(patient_race_df, join_cols, how='left')
  
  return patient_all_data_df


# get encounter table
# Generate encounter table, joining related tables
def get_encounter_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load encounter table
  print("Loading encounter table...")
  table_name = 'encounter' if (table_prefix is None) else '{}_encounter'.format(table_prefix)
  encounter_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in encounter_ceda_table_map.items()])

  # Load encounter admission reason table and aggregate diagnosis fields
  print("Loading encounter admission reason table...")
  table_name = 'encounteradmissionreason' if (table_prefix is None) else '{}_encounteradmissionreason'.format(table_prefix)
  partition_by_cols = ['patient_id', 'encounter_id']
  collect_list_cols = ['admission_reason_diagnosis_id', 'coded_dx', 'free_text_reason']
  w = Window.partitionBy(*partition_by_cols)
  encounter_admission_reason_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in encounter_admission_reason_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()

  # Load encounter chief complaint table and aggregate complaint field
  print("Loading encounter chief complaint table...")
  table_name = 'encounterchiefcomplaint' if (table_prefix is None) else '{}_encounterchiefcomplaint'.format(table_prefix)
  partition_by_cols = ['patient_id', 'encounter_id']
  collect_list_cols = ['chief_complaint']
  w = Window.partitionBy(*partition_by_cols)
  encounter_chief_complaint_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in encounter_chief_complaint_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()

  # Load encounter diagnosis table and aggregate diagnosis fields
  print("Loading encounter diagnosis table...")
  table_name = 'encounterdiagnosis' if (table_prefix is None) else '{}_encounterdiagnosis'.format(table_prefix)
  partition_by_cols = ['patient_id', 'encounter_id']
  collect_list_cols = ['diagnosis_id', 'diagnosis_name', 'primary_diagnosis', 'ed_diagnosis', 'problem_list_id']
  w = Window.partitionBy(*partition_by_cols)
  encounter_diagnosis_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in encounter_diagnosis_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()

  # Join encounter tables
  join_cols = ['patient_id', 'encounter_id']
  encounter_all_data_df = encounter_df \
    .join(encounter_admission_reason_df, join_cols, how='left') \
    .join(encounter_chief_complaint_df, join_cols, how='left') \
    .join(encounter_diagnosis_df, join_cols, how='left')
  encounter_columns = [c for c in encounter_all_data_df.columns if not(c in ['patient_id'])]
  
  # Join patient and encounter tables
  patient_encounter_df = patient_df \
    .join(encounter_all_data_df, ['patient_id'], how='left') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('contact_date') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_contact_dt'),
            F.round((F.unix_timestamp('admission_datetime') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_admission_dt'),
            *encounter_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting encounter records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_encounter_df = patient_encounter_df \
      .where(F.col('contact_date').between(*valid_date_bounds))
  
  return patient_encounter_df


# get problem list table
# Generate problem list table, joining related tables
def get_problem_list_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load problem list table
  print("Loading problem list table...")
  table_name = 'problemlist' if (table_prefix is None) else '{}_problemlist'.format(table_prefix)
  problem_list_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in problem_list_ceda_table_map.items()])
  
  # Load diagnosis table
  print("Loading diagnosis table...")
  table_name = 'diagnosis' if (table_prefix is None) else '{}_diagnosis'.format(table_prefix)
  diagnosis_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in diagnosis_ceda_table_map.items()])
  
  # Join problem list and diagnosis tables
  problem_list_all_data_df = problem_list_df \
    .join(diagnosis_df, ['diagnosis_id'], how='left')
  problem_list_columns = [c for c in problem_list_all_data_df.columns if not(c in ['patient_id'])]
  
  # Join patient and problem list tables
  patient_problem_list_df = patient_df \
    .join(problem_list_all_data_df, ['patient_id'], how='left') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('date_of_entry') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_date_of_entry'),
            *problem_list_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting problem list records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_problem_list_df = patient_problem_list_df \
      .where(F.col('date_of_entry').between(*valid_date_bounds))
  
  return patient_problem_list_df


# get procedure order table
# Generate procedure order table, joining related tables
def get_procedure_order_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load procedure order table
  print("Loading procedure order table...")
  table_name = 'procedureorders' if (table_prefix is None) else '{}_procedureorders'.format(table_prefix)
  procedure_order_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in procedure_order_ceda_table_map.items()])
  procedure_order_columns = [c for c in procedure_order_df.columns if not(c in ['patient_id'])]

  # Load lab result table
  print("Loading lab result table...")
  table_name = 'labresult' if (table_prefix is None) else '{}_labresult'.format(table_prefix)
  lab_result_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in lab_result_ceda_table_map.items()],
            F.when(F.col('unit').isNull(), 'no_unit').otherwise(F.col('unit')).alias('unit_no_null'))
  lab_result_columns = [c for c in lab_result_df.columns if not(c in ['procedure_order_id'])]

  # Join procedure order and lab result tables
  procedure_order_all_data_df = procedure_order_df \
    .join(lab_result_df, ['procedure_order_id'], how='left')
  # Join patient and procedure order tables
  patient_procedure_order_df = patient_df \
    .join(procedure_order_all_data_df, ['patient_id'], how='left') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('ordering_datetime') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_ordering_dt'),
            *procedure_order_columns,
            F.concat_ws('|', F.col('instance'), F.col('common_name'), F.col('result_name'), F.col('unit_no_null')).alias('lab_id'),
            *lab_result_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting procedure order records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_procedure_order_df = patient_procedure_order_df \
      .where(F.col('ordering_datetime').between(*valid_date_bounds))
  
  return patient_procedure_order_df


# get medication order table
# Generate medication order table, joining related tables
def get_medication_order_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load medication order table
  print("Loading medication order table...")
  table_name = 'medicationorders' if (table_prefix is None) else '{}_medicationorders'.format(table_prefix)
  medication_order_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in medication_order_ceda_table_map.items()])

  # Load medication administration table
  print("Loading medication administration table...")
  table_name = 'medicationadministration' if (table_prefix is None) else '{}_medicationadministration'.format(table_prefix)
  medication_administration_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in medication_administration_ceda_table_map.items()])

  # Join medication order and medication administration tables
  medication_order_all_data_df = medication_order_df \
    .join(medication_administration_df, ['patient_id', 'medication_order_id'], how='left')
  medication_order_columns = [c for c in medication_order_all_data_df.columns if not(c in ['patient_id'])]
  
  # Join patient and medication order tables
  patient_medication_order_df = patient_df \
    .join(medication_order_all_data_df, ['patient_id'], how='left') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('ordering_datetime') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_ordering_dt'),
            *medication_order_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting medication order records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_medication_order_df = patient_medication_order_df \
      .where(F.col('ordering_datetime').between(*valid_date_bounds))
  
  return patient_medication_order_df


# get flowsheet table
# Generate flowsheet table, joining related tables
def get_flowsheet_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load flowsheet table
  print("Loading flowsheet table...")
  table_name = 'flowsheet' if (table_prefix is None) else '{}_flowsheet'.format(table_prefix)
  flowsheet_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in flowsheet_ceda_table_map.items()])

  # Load flowsheet entry table
  print("Loading flowsheet entry table...")
  table_name = 'flowsheetentry' if (table_prefix is None) else '{}_flowsheet'.format(table_prefix)
  flowsheet_entry_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in flowsheet_entry_ceda_table_map.items()])

  # Load flowsheet definition table
  print("Loading flowsheet definition table...")
  table_name = 'flowsheetdefinition' if (table_prefix is None) else '{}_flowsheetdefinition'.format(table_prefix)
  flowsheet_definition_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in flowsheet_definition_ceda_table_map.items()])

  # Load inpatient data table
  print("Loading inpatient data table...")
  table_name = 'inpatientdata' if (table_prefix is None) else '{}_inpatientdata'.format(table_prefix)
  inpatient_data_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in inpatient_data_ceda_table_map.items()])
  # Join flowsheet tables
  flowsheet_all_data_df = flowsheet_df \
    .join(flowsheet_entry_df, ['flowsheet_data_id', 'inpatient_data_id'], how='left') \
    .join(flowsheet_definition_df, ['flowsheet_measurement_id'], how='left') \
    .join(inpatient_data_df, ['inpatient_data_id'], how='left')
  flowsheet_columns = [c for c in flowsheet_all_data_df.columns if not(c in ['patient_id'])]

  # Join patient and flowsheet tables
  patient_flowsheet_df = patient_df \
    .join(flowsheet_all_data_df, ['patient_id'], how='left') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('recorded_datetime') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_recorded_dt'),
            *flowsheet_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting flowsheet records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_flowsheet_df = patient_flowsheet_df \
      .where(F.col('recorded_datetime').between(*valid_date_bounds))
  
  return patient_flowsheet_df


# get summary block table
# Generate summary block table, joining related tables
def get_summary_block_table(db='rdp_phi', table_prefix=None, date_bounds=None):
  # Get patient table
  patient_df = get_patient_table(db=db, table_prefix=table_prefix)
  patient_columns = patient_df.columns
  
  # Ensure valid date bounds are provided
  valid_date_bounds = get_valid_date_bounds(date_bounds)
  
  # Load summary block table
  print("Loading summary block table...")
  table_name = 'summaryblock' if (table_prefix is None) else '{}_summaryblock'.format(table_prefix)
  summary_block_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_ceda_table_map.items()])
  
  # Load summary block delivery summary table
  print("Loading summary block delivery summary table...")
  table_name = 'summaryblockdeliverysummary' if (table_prefix is None) else '{}_summaryblockdeliverysummary'.format(table_prefix)
  summary_block_delivery_summary_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_summary_ceda_table_map.items()])
  
  # Load summary block ob history table
  print("Loading summary block ob history table...")
  table_name = 'summaryblockobhistory' if (table_prefix is None) else '{}_summaryblockobhistory'.format(table_prefix)
  summary_block_ob_history_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_ob_history_ceda_table_map.items()])
  
  # Load summary block link table and aggregate fields
  print("Loading summary block link table...")
  table_name = 'summaryblocklink' if (table_prefix is None) else '{}_summaryblocklink'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['ini_encounter_id']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_link_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_link_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block labor and delivery information table and aggregate fields
  print("Loading summary block labor and delivery information table...")
  table_name = 'summaryblocklaboranddeliveryinformation' if (table_prefix is None) else '{}_summaryblocklaboranddeliveryinformation'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['labor_and_delivery_information']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_labor_and_delivery_information_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_labor_and_delivery_information_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block ob delivery signed by table and aggregate fields
  print("Loading summary block ob delivery signed by table...")
  table_name = 'summaryblockobdeliverysignedby' if (table_prefix is None) else '{}_summaryblockobdeliverysignedby'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['ob_delivery_signing_event', 'ob_delivery_signed_time', 'user_id']
  w = Window.partitionBy(*partition_by_cols).orderBy('ob_delivery_signed_time') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  summary_block_ob_delivery_signed_by_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_ob_delivery_signed_by_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block analgesic table and aggregate fields
  print("Loading summary block analgesic table...")
  table_name = 'summaryblockanalgesic' if (table_prefix is None) else '{}_summaryblockanalgesic'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['medication_id']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_analgesic_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_analgesic_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block anesthesia method table and aggregate fields
  print("Loading summary block anesthesia method table...")
  table_name = 'summaryblockanesthesiamethod' if (table_prefix is None) else '{}_summaryblockanesthesiamethod'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['anesthesia_method']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_anesthesia_method_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_anesthesia_method_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery breech type table and aggregate fields
  print("Loading summary block delivery breech type table...")
  table_name = 'summaryblockdeliverybreechtype' if (table_prefix is None) else '{}_summaryblockdeliverybreechtype'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_breech_type']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_breech_type_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_breech_type_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery cord blood disposition table and aggregate fields
  print("Loading summary block delivery cord blood disposition table...")
  table_name = 'summaryblockdeliverycordblooddisposition' if (table_prefix is None) else '{}_summaryblockdeliverycordblooddisposition'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_cord_blood_disposition']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_cord_blood_disposition_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_cord_blood_disposition_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery cord complications table and aggregate fields
  print("Loading summary block delivery cord complications table...")
  table_name = 'summaryblockdeliverycordcomplications' if (table_prefix is None) else '{}_summaryblockdeliverycordcomplications'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_cord_complications']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_cord_complications_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_cord_complications_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery cord vessels table and aggregate fields
  print("Loading summary block delivery cord vessels table...")
  table_name = 'summaryblockdeliverycordvessels' if (table_prefix is None) else '{}_summaryblockdeliverycordvessels'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_cord_vessels']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_cord_vessels_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_cord_vessels_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery forceps location table and aggregate fields
  print("Loading summary block delivery forceps location table...")
  table_name = 'summaryblockdeliveryforcepslocation' if (table_prefix is None) else '{}_summaryblockdeliveryforcepslocation'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_forceps_location']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_forceps_location_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_forceps_location_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery gases sent table and aggregate fields
  print("Loading summary block delivery gases sent table...")
  table_name = 'summaryblockdeliverygasessent' if (table_prefix is None) else '{}_summaryblockdeliverygasessent'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_gases_sent']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_gases_sent_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_gases_sent_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery living status table and aggregate fields
  print("Loading summary block delivery living status table...")
  table_name = 'summaryblockdeliverylivingstatus' if (table_prefix is None) else '{}_summaryblockdeliverylivingstatus'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_living_status']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_living_status_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_living_status_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery presentation type table and aggregate fields
  print("Loading summary block delivery presentation type table...")
  table_name = 'summaryblockdeliverypresentationtype' if (table_prefix is None) else '{}_summaryblockdeliverypresentationtype'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_presentation_type']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_presentation_type_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_presentation_type_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery resuscitation type table and aggregate fields
  print("Loading summary block delivery resuscitation type table...")
  table_name = 'summaryblockdeliveryresuscitationtype' if (table_prefix is None) else '{}_summaryblockdeliveryresuscitationtype'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_resuscitation_type']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_resuscitation_type_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_resuscitation_type_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block delivery vacuum location table and aggregate fields
  print("Loading summary block delivery vacuum location table...")
  table_name = 'summaryblockdeliveryvacuumlocation' if (table_prefix is None) else '{}_summaryblockdeliveryvacuumlocation'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['delivery_vacuum_location']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_delivery_vacuum_location_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_delivery_vacuum_location_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block placenta appearance table and aggregate fields
  print("Loading summary block placenta appearance table...")
  table_name = 'summaryblockplacentaappearance' if (table_prefix is None) else '{}_summaryblockplacentaappearance'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['placenta_appearance']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_placenta_appearance_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_placenta_appearance_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Load summary block placenta removal table and aggregate fields
  print("Loading summary block placenta removal table...")
  table_name = 'summaryblockplacentaremoval' if (table_prefix is None) else '{}_summaryblockplacentaremoval'.format(table_prefix)
  partition_by_cols = ['episode_id']
  collect_list_cols = ['placenta_removal']
  w = Window.partitionBy(*partition_by_cols)
  summary_block_placenta_removal_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .select(*[F.col(v).alias(k) if not(isinstance(v, list)) else F.concat(F.col(v[0]), F.col(v[1])).alias(k) for k, v in summary_block_placenta_removal_ceda_table_map.items()]) \
    .select(*partition_by_cols, *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  # Join summary block tables
  join_cols = ['episode_id']
  summary_block_all_data_df = summary_block_df \
    .join(summary_block_delivery_summary_df, join_cols, how='left') \
    .join(summary_block_ob_history_df, join_cols, how='left') \
    .join(summary_block_link_df, join_cols, how='left') \
    .join(summary_block_labor_and_delivery_information_df, join_cols, how='left') \
    .join(summary_block_ob_delivery_signed_by_df, join_cols, how='left') \
    .join(summary_block_analgesic_df, join_cols, how='left') \
    .join(summary_block_anesthesia_method_df, join_cols, how='left') \
    .join(summary_block_delivery_breech_type_df, join_cols, how='left') \
    .join(summary_block_delivery_cord_blood_disposition_df, join_cols, how='left') \
    .join(summary_block_delivery_cord_complications_df, join_cols, how='left') \
    .join(summary_block_delivery_cord_vessels_df, join_cols, how='left') \
    .join(summary_block_delivery_forceps_location_df, join_cols, how='left') \
    .join(summary_block_delivery_gases_sent_df, join_cols, how='left') \
    .join(summary_block_delivery_living_status_df, join_cols, how='left') \
    .join(summary_block_delivery_presentation_type_df, join_cols, how='left') \
    .join(summary_block_delivery_resuscitation_type_df, join_cols, how='left') \
    .join(summary_block_delivery_vacuum_location_df, join_cols, how='left') \
    .join(summary_block_placenta_appearance_df, join_cols, how='left') \
    .join(summary_block_placenta_removal_df, join_cols, how='left')
  summary_block_columns = [c for c in summary_block_all_data_df.columns if not(c in ['patient_id'])]
  
  # Join patient and summary block tables
  patient_summary_block_df = patient_df \
    .join(summary_block_all_data_df, ['patient_id'], how='inner') \
    .select(*patient_columns,
            F.round((F.unix_timestamp('start_date') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_start_dt'),
            F.round((F.unix_timestamp('labor_onset_datetime') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_labor_onset'),
            F.round((F.unix_timestamp('ob_delivery_delivery_date') - F.unix_timestamp('birth_date'))/86400/365.25, 1).alias('age_at_delivery_dt'),
            (280 + F.round((F.unix_timestamp('ob_delivery_delivery_date') - F.unix_timestamp('working_delivery_date'))/86400, 1)).alias('gestational_days_calculated'),
            *summary_block_columns)
  
  # Get selected records, if valid date bounds are provided
  if not(valid_date_bounds is None):
    print("Limiting summary block records to between '{0}' and '{1}'...".format(*date_bounds))
    patient_summary_block_df = patient_summary_block_df \
      .where(F.col('start_date').between(*valid_date_bounds))
  
  return patient_summary_block_df 

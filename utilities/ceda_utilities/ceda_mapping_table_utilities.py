# Get Diagnosis ID / SNOMED mapping
def get_diagnosis_id_snomed_mapping(omop_table='hadlock_omop_concept'):
  # Load external concept mapping table
  table_name = 'externalconceptmapping'
  external_concept_mapping_df = spark.sql("SELECT * FROM rdp_phi.{}".format(table_name))
  
  # Load OMOP concept table and prepare to join external concepts mapping table
  table_name = omop_table
  omop_concept_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(F.col('vocabulary_id') == 'SNOMED') \
    .withColumnRenamed('vocabulary_id', 'vocabulary') \
    .withColumnRenamed('concept_name', 'omop_description') \
    .select('vocabulary', 'concept_code', 'omop_description')
  
  # Clean columns
  select_cols = ['diagnosis_id', 'vocabulary', 'concept_code', 'name', 'ini']
  external_concept_mapping_cleaned_df = external_concept_mapping_df \
    .withColumn('diagnosis_id', F.concat(F.col('instance'), F.col('value').cast(IntegerType()))) \
    .withColumn('vocabulary_and_code', F.split('concept', '#')) \
    .withColumn('vocabulary', F.col('vocabulary_and_code')[0]) \
    .withColumn('concept_code', F.col('vocabulary_and_code')[1]) \
    .select(select_cols).dropDuplicates()
  
  # Add OMOP concept ids and names
  external_concept_mapping_omop_df = external_concept_mapping_cleaned_df \
    .join(F.broadcast(omop_concept_df), ['vocabulary', 'concept_code'], how='left')
  
  # Specify partition/grouping columns
  partition_by_cols = ['diagnosis_id', 'ini', 'name', 'vocabulary']
  collect_cols = ['concept_code', 'omop_description']
  w = Window.partitionBy(partition_by_cols).orderBy('concept_code') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

  # Group on diagnosis ID
  external_concept_mapping_grouped_df = external_concept_mapping_omop_df \
    .select(*partition_by_cols,
            *[F.collect_list(c).over(w).alias(c) for c in collect_cols]) \
    .dropDuplicates()
  
  return external_concept_mapping_grouped_df


# Get SNOMED / Diagnosis ID map and counts
def get_snomed_diagnosis_id_mapping(
  diagnosis_id_snomed_map='hadlock_diagnosis_id_snomed_map',
  enc_diag_table='hadlock_encounters'):
  
  # Load diagnosis ID / SNOMED map
  diagnosis_id_snomed_map_df = spark.sql(
    "SELECT * FROM rdp_phi_sandbox.{}".format(diagnosis_id_snomed_map)) \
    .where(F.col('ini') == 'EDG')
  
  table_name = enc_diag_table
  enc_diag_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .select(F.explode('diagnosis_id').alias('diagnosis_id')) \
    .groupBy('diagnosis_id').count()
  
  # Join SNOMED and get SNOMED counts
  w = Window.partitionBy('snomed')
  collect_list_cols = ['diagnosis_id', 'omop_description']
  enc_diag_snomed_df = enc_diag_df \
    .join(diagnosis_id_snomed_map_df, ['diagnosis_id'], how='left') \
    .select('diagnosis_id', 'omop_description', 'count',
            F.explode('concept_code').alias('snomed')) \
    .select('snomed', F.sum('count').over(w).alias('count'),
            *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  return enc_diag_snomed_df


# Get Medication ID / RxNorm mapping
def get_medication_id_rxnorm_mapping(omop_table='hadlock_omop_concept'):
  # Load medication rxnorm table
  table_name = 'medicationrxnorm'
  medication_rxnorm_df = spark.sql("SELECT * FROM rdp_phi.{}".format(table_name))

  # Load OMOP concept table and prepare to join medication RxNorm table
  table_name = omop_table
  omop_concept_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(F.col('vocabulary_id') == 'RxNorm') \
    .withColumnRenamed('concept_code', 'rxnorm_code') \
    .withColumnRenamed('concept_name', 'omop_description') \
    .select('rxnorm_code', 'omop_description')

  # Modify columns
  medication_rxnorm_cleaned_df = medication_rxnorm_df \
    .select(F.concat(F.col('instance'), F.col('medication_id')).alias('medication_id'),
            F.col('rxnormcode').alias('rxnorm_code'),
            F.col('codelevel').alias('code_level'),
            F.col('termtype').alias('term_type')) \
    .dropDuplicates()

  # Add OMOP concept ids and names
  medication_rxnorm_omop_df = medication_rxnorm_cleaned_df \
    .join(F.broadcast(omop_concept_df), ['rxnorm_code'], how='left')

  # Specify partition/grouping columns
  partition_by_cols = ['medication_id']
  collect_cols = ['rxnorm_code', 'omop_description', 'code_level', 'term_type']
  w = Window.partitionBy(partition_by_cols).orderBy('rxnorm_code') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

  # Group on medication_id field
  medication_rxnorm_grouped_df = medication_rxnorm_omop_df \
    .select(*partition_by_cols,
            *[F.collect_list(c).over(w).alias(c) for c in collect_cols]) \
    .dropDuplicates()

  # Get medication table
  medication_df = spark.sql("SELECT * FROM rdp_phi.medication") \
    .select(F.concat(F.col('instance'), F.col('medication_id')).alias('medication_id'),
            'name', F.col('shortname').alias('short_name'))

  # Join to the medication table
  medication_rxnorm_final_df = medication_df \
    .join(medication_rxnorm_grouped_df, ['medication_id'], how='left')
  
  return medication_rxnorm_final_df


# Get RxNorm / Medication ID map and counts
def get_rxnorm_medication_id_mapping(
  medication_id_rxnorm_map='hadlock_medication_id_rxnorm_map',
  med_order_table='hadlock_medication_orders',
  omop_table='hadlock_omop_concept'):
  
  # Load medication ID / RxNorm map
  medication_id_rxnorm_map_df = spark.sql(
    "SELECT * FROM rdp_phi_sandbox.{}".format(medication_id_rxnorm_map)) \
    .select('medication_id', F.col('rxnorm_code').alias('mapped_rxnorm_code'), 'omop_description',
            F.col('name').alias('medication_name'), F.lower(F.col('short_name')).alias('short_name'))
  
  # Get counts for medication_id
  table_name = med_order_table
  med_order_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .select('medication_id', 'encounter_id').dropDuplicates() \
    .groupBy('medication_id').count()

  # Get OMOP ingredient codes
  table_name = omop_table
  omop_concept_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(F.col('vocabulary_id') == 'RxNorm') \
    .where(F.col('concept_class_id') == 'Ingredient') \
    .select(F.lower(F.col('concept_name')).alias('ingredient_name'),
            F.col('concept_code').alias('rxnorm_code')) \
    .groupBy('ingredient_name').agg(F.collect_list('rxnorm_code').alias('ingredient_rxnorm_code'))

  # Join RxNorm and get RxNorm counts
  w = Window.partitionBy('rxnorm')
  collect_list_cols = ['medication_id', 'medication_name', 'short_name']
  med_order_rxnorm_df = med_order_df \
    .join(medication_id_rxnorm_map_df, ['medication_id'], how='left') \
    .join(omop_concept_df, F.lower(F.col('medication_name')).startswith(F.col('ingredient_name')), how='left') \
    .withColumn('rxnorm', F.when(F.col('mapped_rxnorm_code').isNotNull(),
                                 F.col('mapped_rxnorm_code')).otherwise(F.col('ingredient_rxnorm_code'))) \
    .select('medication_id', 'medication_name', 'count', 'short_name',
            F.explode('rxnorm').alias('rxnorm')).dropDuplicates() \
    .select('rxnorm', F.sum('count').over(w).alias('count'),
            *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  return med_order_rxnorm_df


# Get Lab ID / LOINC mapping
def get_lab_id_loinc_mapping(labs_table='hadlock_labs', omop_table='hadlock_omop_concept'):
  # Load labs table to create a mapping table
  table_name = labs_table
  collect_cols = ['order_name', 'base_name', 'loinc_code', 'default_loinc_code']
  select_cols = ['lab_id'] + collect_cols
  labs_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(~F.col('lab_id').isin(['1000', '2000', '4000'])) \
    .select(select_cols).dropDuplicates()

  # Load OMOP concept table
  table_name = omop_table
  omop_concept_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(F.col('vocabulary_id') == 'LOINC') \
    .withColumnRenamed('concept_code', 'loinc_code') \
    .withColumnRenamed('concept_name', 'omop_description') \
    .select('loinc_code', 'omop_description')

  # Add OMOP concept names
  labs_omop_df = labs_df \
    .join(F.broadcast(omop_concept_df), ['loinc_code'], how='left')

  # Specify partition/grouping columns
  w = Window.partitionBy('lab_id')
  collect_cols = ['omop_description'] + collect_cols
  labs_loinc_grouped_df = labs_omop_df \
    .select('lab_id', *[F.collect_set(c).over(w).alias(c) for c in collect_cols]) \
    .dropDuplicates()
  
  return labs_loinc_grouped_df


# Get LOINC / Lab ID map and counts
def get_loinc_lab_id_mapping(
  lab_id_loinc_map='hadlock_lab_id_loinc_map',
  labs_table='hadlock_procedure_orders'):

  # Get counts for lab_id
  table_name = labs_table
  labs_grouped_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name)) \
    .where(~F.col('lab_id').isin(['1000', '2000', '4000'])) \
    .groupBy('lab_id').count()

  # Get lab_id to LOINC mapping
  table_name = lab_id_loinc_map
  lab_mapping_df = spark.sql("SELECT * FROM rdp_phi_sandbox.{}".format(table_name))

  # Join LOINC and get LOINC counts
  w = Window.partitionBy('loinc')
  collect_list_cols = ['lab_id', 'omop_description']
  labs_loinc_df = labs_grouped_df \
    .join(lab_mapping_df, ['lab_id'], how='left') \
    .select('lab_id', 'omop_description', 'count', F.explode('loinc_code').alias('loinc')) \
    .select('loinc', F.sum('count').over(w).alias('count'),
            *[F.collect_list(c).over(w).alias(c) for c in collect_list_cols]) \
    .dropDuplicates()
  
  return labs_loinc_df
  
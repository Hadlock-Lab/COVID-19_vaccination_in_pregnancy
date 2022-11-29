# Get infectious disease cohort
def get_infectious_disease_cohort(disease_label, date_bounds=None,
  proc_order_table_name='hadlock_procedure_orders_start_2008',
  db='rdp_phi_sandbox', table_prefix=None):
  
  # Get lab tests within this time period
  current_date = datetime.datetime.now().strftime('%Y-%m-%d')
  date_bounds = ['2008-01-01', current_date] if (date_bounds is None) else get_valid_date_bounds(date_bounds)
  
  # Load procedures orders table and filter by date
  table_name = proc_order_table_name if (table_prefix is None) else '_'.join([str(table_prefix), proc_order_table_name])
  proc_orders_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .where(F.col('ordering_datetime').between(*date_bounds))
  
  # Filter string to identify lab results
  assert(disease_label in list(cc_lab_search_strings.keys()))
  field_search_query_string_list = ['({})'.format(get_field_search_query_string(f, slist, False)) for f, slist in cc_lab_search_strings[disease_label].items()]
  field_search_query_string = ' AND '.join(field_search_query_string_list)

  # Get disease lab results and add cleaned lab result column
  select_col_names = ['patient_id', 'sex', 'ethnic_group', 'age_at_ordering_dt', 'encounter_id', 'ordering_datetime', 'result_value']
  disease_labs_df = proc_orders_df \
    .where(field_search_query_string) \
    .select(select_col_names) \
    .withColumn('cleaned_results', clean_string_udf('result_value')) \
    .withColumn('pcr_naat_outcome', classify_pcr_naat_outcome_udf('result_value', F.lit(disease_label)))
  
  return disease_labs_df


# Maternity cohort columns map
maternity_cohort_columns = {
  'pregnancy_and_delivery_patient_information': ['patient_id', 'pat_id', 'instance', 'birth_date', 'death_date', 'patient_language', 'race', 'ethnic_group', 'sex', 'zip', 'episode_id', 'start_date', 'end_date', 'type', 'ob_delivery_episode_type', 'name', 'status'],
  
  'pregnancy_only_patient_information': ['age_at_delivery_dt', 'age_at_labor_onset', 'age_at_start_dt'],
  
  'delivery_only_patient_information': ['ob_delivery_birth_csn_baby', 'ob_delivery_record_baby_id'],
  
  'coalesce_pregnancy_and_delivery_records': ['gestational_days_calculated', 'ob_hx_gestational_age_days', 'working_delivery_date', 'labor_onset_datetime', 'ob_delivery_labor_onset_date', 'ob_delivery_delivery_date', 'ob_delivery_dilation_complete_date', 'delivery_birth_instant', 'induction_datetime', 'ob_delivery_induction_date', 'ob_delivery_signed_time', 'ob_delivery_signing_event', 'ob_time_rom_to_delivery', 'start_pushing_instant', 'ob_delivery_repair_number_of_packets', 'delivery_placental_instant', 'ob_hx_infant_sex', 'ob_hx_living_status', 'ob_hx_outcome', 'labor_and_delivery_information', 'ob_delivery_delivery_csn_mom', 'feeding_intentions', 'steroids_prior_to_delivery', 'ob_delivery_1st_stage_hours', 'ob_delivery_1st_stage_minutes', 'ob_delivery_2nd_stage_hours', 'ob_delivery_2nd_stage_minutes', 'ob_delivery_3rd_stage_hours', 'ob_delivery_3rd_stage_minutes', 'ob_delivery_augmentation_start_datetime', 'ob_delivery_blood_loss_ml', 'ob_delivery_cervical_ripening_date', 'ob_delivery_decision_time', 'date_of_first_prenatal_care', 'number_of_fetuses', 'ob_delivery_total_delivery_blood_loss_ml', 'ob_expected_delivery_location', 'ob_hx_order', 'ob_reason_for_delivery_location_change', 'ob_source_of_first_prenatal_care', 'ob_sticky_note_text', 'pregravid_bmi', 'pregravid_weight', 'ob_delivery_pregnancy_episode', 'anesthesia_method', 'antibiotics_during_labor', 'delivery_additional_delivery_comments', 'delivery_apgar_1minute', 'delivery_apgar_5minutes', 'delivery_apgar_10minutes', 'delivery_apgar_15minutes', 'delivery_apgar_20minutes', 'delivery_apgar_breathing_1minute', 'delivery_apgar_breathing_5minutes', 'delivery_apgar_breathing_10minutes', 'delivery_apgar_breathing_15minutes', 'delivery_apgar_breathing_20minutes', 'delivery_apgar_grimace_1minute', 'delivery_apgar_grimace_5minutes', 'delivery_apgar_grimace_10minutes', 'delivery_apgar_grimace_15minutes', 'delivery_apgar_grimace_20minutes', 'delivery_apgar_heart_rate_1minute', 'delivery_apgar_heart_rate_5minutes', 'delivery_apgar_heart_rate_10minutes', 'delivery_apgar_heart_rate_15minutes', 'delivery_apgar_heart_rate_20minutes', 'delivery_apgar_muscle_tone_1minute', 'delivery_apgar_muscle_tone_5minutes', 'delivery_apgar_muscle_tone_10minutes', 'delivery_apgar_muscle_tone_15minutes', 'delivery_apgar_muscle_tone_20minutes', 'delivery_apgar_skin_color_1minute', 'delivery_apgar_skin_color_5minutes', 'delivery_apgar_skin_color_10minutes', 'delivery_apgar_skin_color_15minutes', 'delivery_apgar_skin_color_20minutes', 'delivery_birth_comments', 'delivery_breech_type', 'delivery_chest_circumference', 'delivery_cord_blood_disposition', 'delivery_cord_complications', 'delivery_cord_vessels', 'delivery_delivery_method', 'delivery_forceps_location', 'delivery_gases_sent', 'delivery_infant_birth_length_in', 'delivery_infant_birth_weight_oz', 'delivery_infant_head_circumference_in', 'delivery_living_status', 'delivery_living_status_comments', 'delivery_presentation_anterior_posterior', 'delivery_presentation_left_right', 'delivery_presentation_reference', 'delivery_presentation_type', 'delivery_resuscitation_type', 'delivery_vacuum_location', 'ob_delivery_department', 'ob_delivery_labor_identifier', 'ob_delivery_skin_to_skin_end_date', 'ob_delivery_skin_to_skin_start_date', 'ob_history_last_known_living_status', 'ob_hx_delivering_clinician_free_text', 'ob_hx_delivery_site', 'ob_hx_delivery_site_comment', 'ob_hx_outcome_date_confidence', 'ob_hx_total_length_of_labor_hours', 'ob_hx_total_length_of_labor_minutes', 'placenta_appearance', 'placenta_removal', 'prov_id', 'user_id'],
  
  'separate_pregnancy_and_delivery_records': ['ini_encounter_id', 'medication_id', 'curr_loc_id'],
  
  'neither_record_omit': ['anesthesia_patient_reference', 'birth_order', 'csection_trial_of_labor', 'date_of_last_prenatal_care', 'delivery_analgesia_comments', 'delivery_placental_weight', 'delivery_with_forceps_attempted', 'delivery_with_vacuum_attempted', 'month_of_pregnancy_prenatal_care_began', 'number_of_previous_live_births_now_deceased', 'number_of_previous_live_births_now_living', 'ob_delivery_abdominal_girth', 'ob_delivery_birth_order', 'ob_delivery_breast_feeding_initiated_time', 'ob_delivery_complications_comments', 'ob_delivery_contractions_start_date_from_patient', 'ob_delivery_cord_clamping_instant', 'ob_delivery_dilation_start_date', 'ob_delivery_mirrored_record', 'paternity_acknowledgement', 'patient_reported_contractions_start_datetime', 'total_number_of_prenatal_visits']
}


# Get maternity cohort
# Generate maternity cohort
def get_maternity_cohort(summary_block_table_name='hadlock_summary_block_start_2008', db='rdp_phi_sandbox', table_prefix=None):
  # Load summary block table
  table_name = summary_block_table_name if (table_prefix is None) else '_'.join([str(table_prefix), summary_block_table_name])
  pregnancy_types, delivery_types = ['PREGNANCY'], ['DELIVERY', 'DELIVERY SUMMARY', 'STORK CONVERSION HBD']
  summary_block_df = spark.sql("SELECT * FROM {0}.{1}".format(db, table_name)) \
    .where(F.col('type').isin(pregnancy_types + delivery_types))

  # Get pregnancy records
  pregnancy_df = summary_block_df \
    .where(F.col('type').isin(pregnancy_types))

  # Get delivery records
  delivery_df = summary_block_df \
    .where(F.col('type').isin(delivery_types))

  # Select columns from pregnancy records
  pregnancy_df = pregnancy_df \
    .select(*maternity_cohort_columns['pregnancy_and_delivery_patient_information'],
            *maternity_cohort_columns['pregnancy_only_patient_information'],
            *maternity_cohort_columns['coalesce_pregnancy_and_delivery_records'],
            *maternity_cohort_columns['separate_pregnancy_and_delivery_records']) \
    .withColumn('episode_id_join', F.col('episode_id'))

  # Select columns from delivery records
  delivery_df = delivery_df \
    .select(*[F.col(c).alias('{}_child'.format(c)) for c in maternity_cohort_columns['pregnancy_and_delivery_patient_information']],
            *maternity_cohort_columns['delivery_only_patient_information'],
            *[F.col(c).alias('{}_child'.format(c)) for c in maternity_cohort_columns['coalesce_pregnancy_and_delivery_records']],
            *[F.col(c).alias('{}_child'.format(c)) for c in maternity_cohort_columns['separate_pregnancy_and_delivery_records']]) \
    .withColumn('episode_id_join', F.col('ob_delivery_pregnancy_episode_child'))

  # Specify column names
  mother_cols = maternity_cohort_columns['pregnancy_and_delivery_patient_information'] \
    + maternity_cohort_columns['pregnancy_only_patient_information']
  child_cols = ['{}_child'.format(c) for c in maternity_cohort_columns['pregnancy_and_delivery_patient_information']] \
    + maternity_cohort_columns['delivery_only_patient_information']
  coalesce_cols = maternity_cohort_columns['coalesce_pregnancy_and_delivery_records']
  child_coalesce_cols = ['{}_child'.format(c) for c in coalesce_cols]
  pregnancy_and_delivery_cols = maternity_cohort_columns['separate_pregnancy_and_delivery_records']
  pregnancy_and_delivery_cols = [item for pair in zip(pregnancy_and_delivery_cols, ['{}_child'.format(c) for c in pregnancy_and_delivery_cols]) for item in pair]

  # Join pregnancy (mother) and delivery (child) records
  maternity_cohort_df = pregnancy_df \
    .join(delivery_df, ['episode_id_join'], how='left') \
    .select(*mother_cols, *child_cols,
            *[F.coalesce(c1, c2).alias(c1) for c1, c2 in zip(coalesce_cols, child_coalesce_cols)],
            *pregnancy_and_delivery_cols) \
    .select(*mother_cols,
            F.coalesce(280 + F.datediff('ob_delivery_delivery_date', 'working_delivery_date'), 'ob_hx_gestational_age_days').alias('gestational_days'),
            *child_cols, *coalesce_cols, *pregnancy_and_delivery_cols)
  
  return maternity_cohort_df
  
# Patient table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
patient_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'pat_id': 'pat_id',
  'instance': 'instance',
  'birth_date': 'birthdate',
  'death_date': 'deathdate',
  'sex': 'sex',
  'ethnic_group': 'ethnicgroup',
  'patient_language': 'patientlanguage',
  'zip': 'zip',
  'curr_loc_id': ['instance', 'curr_loc_id']
}


# Patient race table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
patient_race_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'race': 'race'
}


# Encounter table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
encounter_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'pat_enc_csn_id': 'pat_enc_csn_id',
  'contact_date': 'contact_date',
  'department_id': ['instance', 'department_id'],
  'facility_state': 'facilitystate',
  'facility_zip': 'facilityzip',
  'mode_of_arrival': 'modeofarrival',
  'acuity': 'acuity',
  'accomodation_code': 'accomodationcode',
  'admission_datetime': 'admissiondatetime',
  'discharge_datetime': 'dischargedatetime',
  'arrival_datetime': 'arrivaldatetime',
  'patient_status': 'patientstatus',
  'confirmation_status': 'confirmationstatus',
  'ed_disposition': 'eddisposition',
  'discharge_disposition': 'dischargedisposition',
  'patient_class': 'patientclass',
  'discharge_destination': 'dischargedestination',
  'expected_discharge_date': 'expecteddischargedate',
  'discharge_transportation': 'dischargetransportation',
  'encounter_type': 'encountertype',
  'cancel_reason': 'cancelreason',
  'appointment_datetime': 'appointmentdatetime',
  'appointment_status': 'appointmentstatus',
  'visit_type': 'visittype',
  'expected_admission_date': 'expectedadmissiondate',
  'obstetric_status': 'obstetricstatus',
  'bp_systolic': 'bp_systolic',
  'bp_diastolic': 'bp_diastolic',
  'temperature': 'temperature',
  'pulse': 'pulse',
  'weight': 'weight',
  'height': 'height',
  'respirations': 'respirations',
  'bmi': 'bmi',
  'hsp_account_id': ['instance', 'hsp_account_id'],
  'ed_episode_id': ['instance', 'ed_episode_id'],
  'ip_episode_id': ['instance', 'ip_episode_id'],
  'visit_prov_id': ['instance', 'visit_prov_id'],
  'discharge_prov_id': ['instance', 'discharge_prov_id'],
  'admission_prov_id': ['instance', 'admission_prov_id'],
  'bill_attend_prov_id': ['instance', 'bill_attend_prov_id'],
  'pri_problem_id': ['instance', 'pri_problem_id']
}


# Encounter admission reason table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
encounter_admission_reason_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'admission_reason_diagnosis_id': ['instance', 'dx_id'],
  'coded_dx': 'codeddx',
  'free_text_reason': 'freetextreason'
}


# Encounter chief complaint table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
encounter_chief_complaint_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'chief_complaint': 'chiefcomplaint'
}


# Encounter diagnosis table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
encounter_diagnosis_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'diagnosis_id': ['instance', 'dx_id'],
  'diagnosis_name': 'diagnosisname',
  'primary_diagnosis': 'primarydiagnosis',
  'ed_diagnosis': 'eddiagnosis',
  'problem_list_id': ['instance', 'problem_list_id']
}


# Problem list table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
problem_list_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'noted_date': 'noted_date',
  'date_of_entry': 'date_of_entry',
  'resolved_date': 'resolved_date',
  'problem_status': 'problemstatus',
  'chronic_yn': 'chronic_yn',
  'diagnosis_id': ['instance', 'dx_id'],
  'problem_list_id': ['instance', 'problem_list_id'],
  'overview_note_id': ['instance', 'overview_note_id'],
  'creating_order_id': ['instance', 'creating_order_id']  
}


# Diagnosis table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
diagnosis_ceda_table_map = {
  'diagnosis_id': ['instance', 'dx_id'],
  'diagnosis_name': 'name'
}


# Procedure order table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
procedure_order_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'procedure_order_id': ['instance', 'order_proc_id'],
  'order_name': 'ordername',
  'order_description': 'orderdescription',
  'order_class': 'orderclass',
  'order_mode': 'ordermode',
  'order_priority': 'orderpriority',
  'order_status': 'orderstatus',
  'order_type': 'ordertype',
  'ordering_datetime': 'orderingdatetime',
  'abnormal_yn': 'abnormal_yn',
  'category': 'category',
  'authorizing_prov_id': ['instance', 'authrzing_prov_id'],
  'order_set': 'orderset',
  'future_or_stand': 'future_or_stand',
  'referral_class': 'referralclass',
  'referral_priority': 'referralpriority',
  'referral_type': 'referraltype',
  'referral_reason': 'referralreason',
  'referral_number_of_visits': 'referralnumberofvisits',
  'referral_expiration_date': 'referralexpirationdate',
  'referred_to_specialty': 'referredtospecialty',
  'parent_order_id': ['instance', 'parent_order_id'],
  'referred_to_department_id': ['instance', 'referredtodepartmentid'],
  'referred_to_facility_id': ['instance', 'referredtofacilityid'],
  'department_id': ['instance', 'department_id']
}


# Lab result table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
lab_result_ceda_table_map = {
  'procedure_order_id': ['instance', 'order_proc_id'],
  'lab_encounter_id': ['instance', 'pat_enc_csn_id'],
  'lab_order_description': 'orderdescription',
  'result_value': 'resultvalue',
  'result_num_value': 'resultnumvalue',
  'flagged_as': 'flaggedas',
  'low_threshold': 'lowthreshold',
  'high_threshold': 'highthreshold',
  'unit': 'unit',
  'status': 'status',
  'type': 'type',
  'observation_datetime': 'observationdatetime',
  'component_loinc_id': 'compon_lnc_id',
  'loinc_code': 'loinccode',
  'organism_quantity': 'organismquantity',
  'organism_quantity_unit': 'organismquantityunit',
  'organism_id': ['instance', 'organism_id'],
  'organism_name': 'organismname',
  'result_name': 'resultname',
  'base_name': 'basename',
  'common_name': 'commonname',
  'default_loinc_code': 'defaultloinccode',
  'default_loinc_id': 'default_lnc_id',
  'snomed': 'snomed',
  'result_comment': 'resultcomment',
  'result_datetime': 'resultdatetime',
  'full_addl_result_comment': 'fulladdlresultcomment'
}


# Medication order table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
medication_order_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'medication_order_id': ['instance', 'order_med_id'],
  'medication_id': ['instance', 'medication_id'],
  'order_description': 'orderdescription',
  'order_class': 'orderclass',
  'order_mode': 'ordermode',
  'order_set': 'orderset',
  'order_priority': 'orderpriority',
  'order_status': 'orderstatus',
  'ordering_datetime': 'orderingdatetime',
  'start_date': 'start_date',
  'end_date': 'end_date',
  'sig': 'sig',
  'dosage': 'dosage',
  'quantity': 'quantity',
  'route': 'route',
  'frequency_name': 'freq_name',
  'number_of_times': 'number_of_times',
  'time_unit': 'time_unit',
  'prn_yn': 'prn_yn',
  'department_id': ['instance', 'department_id'],
  'authorizing_prov_id': ['instance', 'authrzing_prov_id'],
  'order_prov_id': ['instance', 'ord_prov_id']  
}


# Medication administration table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
medication_administration_ceda_table_map = {
  'medication_order_id': ['instance', 'order_med_id'],
  'patient_id': ['instance', 'pat_id'],
  'administration_datetime': 'administrationdatetime',
  'action_taken': 'actiontaken',
  'recorded_datetime': 'recordeddatetime',
  'due_datetime': 'duedatetime',
  'is_timely': 'istimely',
  'scheduled_for_datetime': 'scheduledfordatetime',
  'scheduled_on_datetime': 'scheduledondatetime',
  'med_admin_department_id': ['instance', 'department_id']
}


# Flowsheet table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
flowsheet_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'flowsheet_data_id': ['instance', 'fsd_id'],
  'inpatient_data_id': ['instance', 'inpatient_data_id']
}


# Flowsheet entry table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
flowsheet_entry_ceda_table_map = {
  'flowsheet_data_id': ['instance', 'fsd_id'],
  'inpatient_data_id': ['instance', 'inpatient_data_id'],
  'recorded_datetime': 'recorded_time',
  'flowsheet_measurement_id': ['instance', 'flo_meas_id'],
  'value': 'value'
}


# Flowsheet definition table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
flowsheet_definition_ceda_table_map = {
  'flowsheet_measurement_id': ['instance', 'flo_meas_id'],
  'name': 'name',
  'display_name': 'displayname',
  'type': 'type',
  'value_type': 'valuetype'
}


# Inpatient data table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
inpatient_data_ceda_table_map = {
  'inpatient_data_id': ['instance', 'inpatient_data_id'],
  'encounter_id': ['instance', 'pat_enc_csn_id'],
  'status': 'status'
}


# Summary block table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_ceda_table_map = {
  'patient_id': ['instance', 'pat_id'],
  'episode_id': ['instance', 'episode_id'],
  'status': 'status',
  'name': 'name',
  'start_date': 'startdate',
  'end_date': 'enddate',
  'type': 'type',
  'working_delivery_date': 'workingdeliverydate',
  'number_of_fetuses': 'numberoffetuses',
  'labor_onset_datetime': 'laboronsetdatetime',
  'pregravid_weight': 'pregravidweight',
  'pregravid_bmi': 'pregravidbmi',
  'birth_order': 'birthorder',
  'patient_reported_contractions_start_datetime': 'patientreportedcontractionsstartdatetime',
  'induction_datetime': 'inductiondatetime',
  'ob_sticky_note_text': 'obstickynotetext',
  'anesthesia_patient_reference': 'anesthesiapatientreference'
}


# Summary block delivery summary table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_summary_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'ob_delivery_delivery_csn_mom': ['instance', 'obdeliverydeliverycsnmom'],
  'ob_delivery_birth_csn_baby': ['instance', 'obdeliverybirthcsn_baby'],
  'antibiotics_during_labor': 'antibioticsduringlabor',
  'csection_trial_of_labor': 'csectiontrialoflabor',
  'delivery_additional_delivery_comments': 'deliveryadditionaldeliverycomments',
  'delivery_analgesia_comments': 'deliveryanalgesiacomments',
  'delivery_birth_comments': 'deliverybirthcomments',
  'delivery_birth_instant': 'deliverybirthinstant',
  'delivery_chest_circumference': 'deliverychestcircumference',
  'delivery_delivery_method': 'deliverydeliverymethod',
  'delivery_infant_birth_length_in': 'deliveryinfantbirthlengthin',
  'delivery_infant_birth_weight_oz': 'deliveryinfantbirthweightoz',
  'delivery_infant_head_circumference_in': 'deliveryinfantheadcircumferencein',
  'delivery_living_status_comments': 'deliverylivingstatuscomments',
  'delivery_placental_instant': 'deliveryplacentalinstant',
  'delivery_placental_weight': 'deliveryplacentalweight',
  'delivery_presentation_anterior_posterior': 'deliverypresentationanteriorposterior',
  'delivery_presentation_left_right': 'deliverypresentationleftright',
  'delivery_presentation_reference': 'deliverypresentationreference',
  'delivery_with_forceps_attempted': 'deliverywithforcepsattempted',
  'delivery_with_vacuum_attempted': 'deliverywithvacuumattempted',
  'ob_delivery_1st_stage_hours': 'obdelivery1ststagehours',
  'ob_delivery_1st_stage_minutes': 'obdelivery1ststageminutes',
  'ob_delivery_2nd_stage_hours': 'obdelivery2ndstagehours',
  'ob_delivery_2nd_stage_minutes': 'obdelivery2ndstageminutes',
  'ob_delivery_3rd_stage_hours': 'obdelivery3rdstage_hours',
  'ob_delivery_3rd_stage_minutes': 'obdelivery3rdstageminutes',
  'ob_delivery_abdominal_girth': 'obdeliveryabdominalgirth',
  'ob_delivery_augmentation_start_datetime': 'obdeliveryaugmentationstartdatetime',
  'ob_delivery_birth_order': 'obdeliverybirthorder',
  'ob_delivery_blood_loss_ml': 'obdeliverybloodlossml',
  'ob_delivery_breast_feeding_initiated_time': 'obdeliverybreastfeedinginitiatedtime',
  'ob_delivery_cervical_ripening_date': 'obdeliverycervicalripeningdate',
  'ob_delivery_complications_comments': 'obdeliverycomplicationscomments',
  'ob_delivery_contractions_start_date_from_patient': 'obdeliverycontractionsstartdatefrompatient',
  'ob_delivery_cord_clamping_instant': 'obdeliverycordclampinginstant',
  'ob_delivery_decision_time': 'obdeliverydecisiontime',
  'ob_delivery_delivery_date': 'obdeliverydeliverydate',
  'ob_delivery_dilation_start_date': 'obdeliverydilationstartdate',
  'ob_delivery_dilation_complete_date': 'obdeliverydilationcompletedate',
  'ob_delivery_episode_type': 'obdeliveryepisodetype',
  'ob_delivery_induction_date': 'obdeliveryinductiondate',
  'ob_delivery_labor_identifier': 'obdeliverylaboridentifier',
  'ob_delivery_labor_onset_date': 'obdeliverylaboronsetdate',
  'ob_delivery_mirrored_record': 'obdeliverymirroredrecord',
  'ob_delivery_pregnancy_episode': ['instance', 'obdeliverypregnancyepisode'],
  'ob_delivery_record_baby_id': ['instance', 'obdeliveryrecordbabyid'],
  'ob_delivery_repair_number_of_packets': 'obdeliveryrepairnumberofpackets',
  'ob_delivery_skin_to_skin_start_date': 'obdeliveryskintoskinstartdate',
  'ob_delivery_skin_to_skin_end_date': 'obdeliveryskintoskinenddate',
  'ob_delivery_total_delivery_blood_loss_ml': 'obdeliverytotaldeliverybloodlossml',
  'ob_expected_delivery_location': ['instance', 'obexpecteddeliverylocation'],
  'ob_reason_for_delivery_location_change': 'obreasonfordeliverylocationchange',
  'ob_time_rom_to_delivery': 'obtimeromtodelivery',
  'start_pushing_instant': 'startpushinginstant',
  'steroids_prior_to_delivery': 'steroidspriortodelivery',
  'delivery_apgar_1minute': 'deliveryapgar1minute',
  'delivery_apgar_5minutes': 'deliveryapgar5minute',
  'delivery_apgar_10minutes': 'deliveryapgar10minute',
  'delivery_apgar_15minutes': 'deliveryapgar15minute',
  'delivery_apgar_20minutes': 'deliveryapgar20minute',
  'delivery_apgar_breathing_1minute': 'deliveryapgarbreathing1minute',
  'delivery_apgar_breathing_5minutes': 'deliveryapgarbreathing5minutes',
  'delivery_apgar_breathing_10minutes': 'deliveryapgarbreathing10minutes',
  'delivery_apgar_breathing_15minutes': 'deliveryapgarbreathing15minutes',
  'delivery_apgar_breathing_20minutes': 'deliveryapgarbreathing20minutes',
  'delivery_apgar_grimace_1minute': 'deliveryapgargrimace1minute',
  'delivery_apgar_grimace_5minutes': 'deliveryapgargrimace5minutes',
  'delivery_apgar_grimace_10minutes': 'deliveryapgargrimace10minutes',
  'delivery_apgar_grimace_15minutes': 'deliveryapgargrimace15minutes',
  'delivery_apgar_grimace_20minutes': 'deliveryapgargrimace20minutes',
  'delivery_apgar_heart_rate_1minute': 'deliveryapgarheartrate1minute',
  'delivery_apgar_heart_rate_5minutes': 'deliveryapgarheartrate5minutes',
  'delivery_apgar_heart_rate_10minutes': 'deliveryapgarheartrate10minutes',
  'delivery_apgar_heart_rate_15minutes': 'deliveryapgarheartrate15minutes',
  'delivery_apgar_heart_rate_20minutes': 'deliveryapgarheartrate20minutes',
  'delivery_apgar_muscle_tone_1minute': 'deliveryapgarmuscletone1minute',
  'delivery_apgar_muscle_tone_5minutes': 'deliveryapgarmuscletone5minutes',
  'delivery_apgar_muscle_tone_10minutes': 'deliveryapgarmuscletone10minutes',
  'delivery_apgar_muscle_tone_15minutes': 'deliveryapgarmuscletone15minutes',
  'delivery_apgar_muscle_tone_20minutes': 'deliveryapgarmuscletone20minutes',
  'delivery_apgar_skin_color_1minute': 'deliveryapgarskincolor1minute',
  'delivery_apgar_skin_color_5minutes': 'deliveryapgarskincolor5minutes',
  'delivery_apgar_skin_color_10minutes': 'deliveryapgarskincolor10minutes',
  'delivery_apgar_skin_color_15minutes': 'deliveryapgarskincolor15minutes',
  'delivery_apgar_skin_color_20minutes': 'deliveryapgarskincolor20minutes',
  'ob_delivery_department': ['instance', 'obdeliverydepartment'],
  'prov_id': ['instance', 'prov_id']
}



# Summary block OB history table column map

# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_ob_history_ceda_table_map = {
  'episode_id': ['instance', 'episode_id'],
  'date_of_first_prenatal_care': 'dateoffirstprenatalcare',
  'date_of_last_prenatal_care': 'dateoflastprenatalcare',
  'feeding_intentions': 'feedingintentions',
  'month_of_pregnancy_prenatal_care_began': 'monthofpregnancyprenatalcarebegan',
  'number_of_previous_live_births_now_deceased': 'numberofpreviouslivebirthsnowdeceased',
  'number_of_previous_live_births_now_living': 'numberofpreviouslivebirthsnowliving',
  'ob_source_of_first_prenatal_care': 'obsourceoffirstprenatalcare',
  'paternity_acknowledgement': 'paternityacknowledgement',
  'total_number_of_prenatal_visits': 'totalnumberofprenatalvisits',
  'ob_hx_living_status': 'obhxlivingstatus',
  'ob_history_last_known_living_status': 'obhistorylastknownlivingstatus',
  'ob_hx_delivering_clinician_free_text': 'obhxdeliveringclinicianfreetext',
  'ob_hx_delivery_site': 'obhxdeliverysite',
  'ob_hx_delivery_site_comment': 'obhxdeliverysitecomment',
  'ob_hx_gestational_age_days': 'obhxgestationalagedays',
  'ob_hx_infant_sex': 'obhxinfantsex',
  'ob_hx_order': 'obhxorder',
  'ob_hx_outcome': 'obhxoutcome',
  'ob_hx_outcome_date_confidence': 'obhxoutcomedateconfidence',
  'ob_hx_total_length_of_labor_hours': 'obhxtotallengthoflaborhours',
  'ob_hx_total_length_of_labor_minutes': 'obhxtotallengthoflaborminutes'
}

# Note: The following fields are all null as of 2021-03-06
# fatherbirthcontinent
# fatherbirthcountry
# fatheroccupation
# fathersbirthplace
# fatherscity
# fathersdateofbirth
# fatherseducation
# fathersethnicity
# fathersname
# fathersstate
# fatherszipcode
# insidecitylimits
# motherbirthplace
# motherdrinksperweek3monthsbeforepregnancy
# motherdrinksperweekfirst3monthsofpregnancy
# motherdrinksperweeksecond3monthsofpregnancy
# motherdrinksperweekthirdtrimesterofpregnancy
# mothermarriedconceptiontobirth
# mothersmoking3monthsbeforepregnancyquantity
# mothersmoking3monthsbeforepregnancyunit
# mothersmokingfirst3monthsofpregnancyquantity
# mothersmokingfirst3monthsofpregnancyunit
# mothersmokingsecond3monthsofpregnancyquantity
# mothersmokingsecond3monthsofpregnancyunit
# mothersmokingthirdtrimesterofpregnancyquantity
# mothersmokingthirdtrimesterofpregnancyunit
# wicfood


# Summary block link table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_link_ceda_table_map = {
  'episode_id': ['instance', 'episode_id'],
  'ini_encounter_id': ['instance', 'inicsn']
}


# Summary block labor and delivery information table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_labor_and_delivery_information_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'labor_and_delivery_information': 'laboranddeliveryinformation'
}

# Summary block ob delivery signed by table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_ob_delivery_signed_by_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'ob_delivery_signing_event': 'obdeliverysigningevent',
  'ob_delivery_signed_time': 'obdeliverysignedtime',
  'user_id': ['instance', 'user_id']
}


# Summary block analgesic table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_analgesic_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'medication_id': ['instance', 'medication_id']
}


# Summary block anesthesia method table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_anesthesia_method_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'anesthesia_method': 'anesthesiamethod'
}


# Summary block delivery breech type table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_breech_type_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_breech_type': 'deliverybreechtype'
}


# Summary block delivery cord blood disposition table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_cord_blood_disposition_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_cord_blood_disposition': 'deliverycordblooddisposition'
}


# Summary block delivery cord complications table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_cord_complications_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_cord_complications': 'deliverycordcomplications'
}


# Summary block delivery cord vessels table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_cord_vessels_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_cord_vessels': 'deliverycordvessels'
}


# Summary block delivery forceps location table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_forceps_location_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_forceps_location': 'deliveryforcepslocation'
}


# Summary block delivery gases sent table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_gases_sent_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_gases_sent': 'deliverygasessent'
}


# Summary block delivery living status table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_living_status_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_living_status': 'deliverylivingstatus'
}


# Summary block delivery presentation type table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_presentation_type_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_presentation_type': 'deliverypresentationtype'
}


# Summary block delivery resuscitation type table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_resuscitation_type_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_resuscitation_type': 'deliveryresuscitationtype'
}


# Summary block delivery vacuum location table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_delivery_vacuum_location_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'delivery_vacuum_location': 'deliveryvacuumlocation'
}


# Summary block placenta appearance table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_placenta_appearance_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'placenta_appearance': 'placentaappearance'
}


# Summary block placenta removal table column map
# Specify CEDA table column names and what ReDaP column(s) they correspond to
# If multiple columns are specified, then concatenate columns
summary_block_placenta_removal_ceda_table_map = {
  'episode_id': ['instance', 'summary_block_id'],
  'placenta_removal': 'placentaremoval'
}



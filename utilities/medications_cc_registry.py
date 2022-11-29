# A02A: Antacids
# Magnesium compounds
antacid_A02AA_codes = [
  '29155',    # magnesium carbonate
  '6581',     # magnesium hydroxide
  '6582',     # magnesium oxide
  '1666106',  # magnesium peroxide
  '1310566'   # magnesium silicate
]

# Aluminium compounds
antacid_A02AB_codes = [
  '17392',  # aloglutamol
  '612',    # aluminum hydroxide
  '17618'   # aluminum phosphate
]

# Calcium compounds
antacid_A02AC_codes = [
  '1897',     # calcium carbonate
  '1311526'   # calcium silicate
]

antacid_codes = antacid_A02AA_codes + antacid_A02AB_codes + antacid_A02AC_codes


# A02B: Drugs for peptic ulcer and gastroesophageal reflux disease (GERD)
# H2-receptor antagonists
peptic_ulcer_gerd_A02BA_codes = [
  '2541',    # cimetidine
  '4278',    # famotidine
  '42319',   # nizatidine
  '9143',    # ranitidine
  '114817'   # roxatidine
]

# Proton pump inhibitors
peptic_ulcer_gerd_A02BC_codes = [
  '816346,'  # dexlansoprazole
  '283742',  # esomeprazole
  '17128',   # lansoprazole
  '7646',    # omeprazole
  '40790',   # pantoprazole
  '114979'   # rabeprazole
]

# Other drugs for peptic ulcer and gastro-oesophageal reflux disease (GORD)
peptic_ulcer_gerd_A02BX_codes = [
  '16784',   # acetoxolone
  '17305',   # alginic acid
  '47181',   # bismuth subcitrate
  '19477',   # bismuth subnitrate
  '2017',    # carbenoxolone
  '8352',    # pirenzepine
  '8730',    # proglumide
  '10156'    # sucralfate
]

peptic_ulcer_gerd_codes = peptic_ulcer_gerd_A02BA_codes + peptic_ulcer_gerd_A02BC_codes + peptic_ulcer_gerd_A02BX_codes


# C01CA: Adrenergic and dopaminergic agents
adrenergic_dopaminergic_codes = [
  '61609',    # arbutamine
  '47589',    # cafedrine
  '3616',     # dobutamine
  '3628',     # dopamine
  '23638',    # dopexamine
  '1489913',  # droxidopa
  '3966',     # ephedrine
  '3992',     # epinephrine
  '4169',     # etilefrine
  '24853',    # fenoldopam
  '41157',    # gepefrine
  '51253',    # ibopamine
  '6054',     # isoproterenol
  '6756',     # mephentermine
  '6805',     # metaraminol
  '6853',     # methoxamine
  '6963',     # midodrine
  '7512',     # norepinephrine
  '31988',    # norfenefrine
  '1311141',  # octopamine
  '8163',     # phenylephrine
  '31958'     # theodrenaline
]


# Anticoagulants
heparin_codes = [
  '5224',    # heparin
  '1009',    # antithrombin III
  '67109',   # dalteparin
  '67108',   # enoxaparin
  '67031',   # nadroparin
  '69528',   # parnaparin
  '75960',   # reviparin
  '78484',   # danaparoid
  '69646',   # tinzaparin
  '280611'   # bemiparin
]

# Vitamin K antagonists, Direct thrombin inhibitors, Direct factor Xa inhibitors
other_anticoagulant_codes = [
  "1598",    # dicumarol
  "8130",    # phenindione
  "11289",   # warfarin
  "8150",    # phenprocoumon
  "154",     # acenocoumarol
  "163426",  # tioclomarol
  "50097",   # fluindione
  "237057",  # lepirudin
  "1037042", # dabigatran etexilate
  "1114195", # rivaroxaban
  "1364430", # apixaban
  "1599538", # edoxaban
  "1927851", # betrixaban
  "15202",   # argatroban
  "60819",   # bivalirudin
  "114934",  # desirudin
  "237057",  # lepirudin
  "321208"   # fondaparinux
]

anticoagulant_codes = heparin_codes + other_anticoagulant_codes


# C02: Antihypertensives
# Antiadrenergic agents, centrally acting
antihypertensive_C02A_codes = [
  '2599',    # clonidine
  '62174',   # deserpidine
  '40114',   # guanfacine
  '235829',  # methoserpidine
  '6876',    # methyldopa
  '30257',   # moxonidine
  '9259',    # rescinnamine
  '9260',    # reserpine
  '55679'    # rilmenidine
]

# Antiadrenergic agents, ganglion-blocking
antihypertensive_C02B_codes = [
  '6673',  # mecamylamine
  '10828'  # trimethaphan
]

# Antiadrenergic agents, peripherally acting
antihypertensive_C02C_codes = [
  '1523',   # bethanidine
  '3118',   # debrisoquin
  '49276',  # doxazosin
  '5036',   # guanethidine
  '5784',   # indoramin
  '8629',   # prazosin
  '39230'   # urapidil
]

# Agents acting on arteriolar smooth muscle
antihypertensive_C02D_codes = [
  '3327',  # diazoxide
  '3409',  # dihydralazine
  '5470',  # hydralazine
  '6984',  # minoxidil
  '7476',  # nitroprusside
  '33717'  # pinacidil
]

# Other antihypertensives
antihypertensive_C02K_codes = [
  '358274',   # ambrisentan
  '75207',    # bosentan
  '6131',     # ketanserin
  '1442132',  # macitentan
  '266604',   # metyrosine
  '7930',     # pargyline
  '1439816'   # riociguat
]

antihypertensive_codes = antihypertensive_C02A_codes + antihypertensive_C02B_codes + antihypertensive_C02C_codes + antihypertensive_C02D_codes + antihypertensive_C02K_codes


# C07A: Beta blocking agents
beta_blocker_codes = [
  '149',    # acebutolol
  '597',    # alprenolol
  '1202',   # atenolol
  '1520',   # betaxolol
  '19484',  # bisoprolol
  '19605',  # bopindolol
  '1817',   # bupranolol
  '2116',   # carteolol
  '20352',  # carvedilol
  '20498',  # celiprolol
  '49737',  # esmolol
  '6185',   # labetalol
  '29518',  # mepindolol
  '6918',   # metoprolol
  '7226',   # nadolol
  '31555',  # nebivolol
  '7801',   # oxprenolol
  '7973',   # penbutolol
  '8332',   # pindolol
  '8620',   # practolol
  '8787',   # propranolol
  '9947',   # sotalol
  '37546',  # talinolol
  '37840',  # tertatolol
  '10600'   # timolol
]


# C08: Calcium channel blockers
# Selective calcium channel blockers with mainly vascular effects
calcium_channel_blocker_C08C_codes = [
  '17767',   # amlodipine
  '233603',  # clevidipine
  '4316',    # felodipine
  '33910',   # isradipine
  '28382',   # lacidipine
  '135056',  # lercanidipine
  '29275',   # manidipine
  '39879',   # mepirodipine
  '83213',   # mibefradil
  '7396',    # nicardipine
  '7417',    # nifedipine
  '53692',   # nilvadipine
  '7426',    # nimodipine
  '7435',    # nisoldipine
  '7441'     # nitrendipine
]

# Selective calcium channel blockers with direct cardiac effects
calcium_channel_blocker_C08D_codes = [
  '3443',  # diltiazem
  '4648',  # gallopamil
  '11170'  # verapamil
]

# Non-selective calcium channel blockers
calcium_channel_blocker_C08E_codes = [
  '1436',   # bepridil
  '4327',   # fendiline
  '6390',   # lidoflazine
  '8050'    # perhexiline
]

calcium_channel_blocker_codes = calcium_channel_blocker_C08C_codes + calcium_channel_blocker_C08D_codes + calcium_channel_blocker_C08E_codes


# C09: Agents acting on the renin-angiotensin system
# ACE inhibitors, plain
renin_angiotensin_C09A_codes = [
  '18867',  # benazepril
  '1998',   # captopril
  '21102',  # cilazapril
  '3827',   # enalapril
  '50166',  # fosinopril
  '60245',  # imidapril
  '29046',  # lisinopril
  '30131',  # moexipril
  '54552',  # perindopril
  '35208',  # quinapril
  '35296',  # ramipril
  '36908',  # spirapril
  '38454',  # trandolapril
  '39990'   # zofenopril
]

# Angiotensin II receptor blockers (ARBs)
renin_angiotensin_C09C_codes = [
  '1091643',  # azilsartan
  '214354',   # candesartan
  '83515',    # eprosartan
  '83818',    # irbesartan
  '52175',    # losartan
  '321064',   # olmesartan
  '73494',    # telmisartan
  '69749'     # valsartan
]

# Other agents acting on the renin-angiotensin system
renin_angiotensin_C09X_codes = [
  '325646'   # aliskiren
]

renin_angiotensin_codes = renin_angiotensin_C09A_codes + renin_angiotensin_C09C_codes + renin_angiotensin_C09X_codes


# N06: Antidepressants
# Non-selective monoamine reuptake inhibitors
antidepressant_N06AA_codes = [
  '17698',    # amineptin
  '704',      # amitriptyline
  '722',      # amoxapine
  '19895',    # butriptyline
  '2597',     # clomipramine
  '3247',     # desipramine
  '3332',     # dibenzepin
  '3634',     # dothiepin
  '3638',     # doxepin
  '5691',     # imipramine
  '5979',     # iprindole
  '6465',     # lofepramine
  '6646',     # maprotiline
  '446248',   # melitracen
  '7531',     # nortriptyline
  '7674',     # opipramol
  '8886',     # protriptyline
  '35242',    # quinupramine
  '10834'     # trimipramine
]

# Selective serotonin reuptake inhibitors
antidepressant_N06AB_codes = [
  '2556',     # citalopram
  '321988',   # escitalopram
  '4493',     # fluoxetine
  '42355',    # fluvoxamine
  '32937',    # paroxetine
  '36437'     # sertraline
]

# Monoamine oxidase inhibitors, non-selective
antidepressant_N06AF_codes = [
  '6011',   # isocarboxazid
  '7394',   # nialamide
  '8123',   # phenelzine
  '10734'   # tranylcypromine
]

# Monoamine oxidase A inhibitors
antidepressant_N06AG_codes = [
  '30121',   # moclobemide
  '38382'    # toloxatone
]

# Monoamine oxidase B inhibitors (add this for antidepressant?)
# '9639'  # selegiline

# Other antidepressants
antidepressant_N06AX_codes = [
  '94',        # 5-hydroxytryptophan
  '47111',     # bifemelane
  '42347',     # bupropion
  '734064',    # desvenlafaxine
  '72625',     # duloxetine
  '2119365',   # esketamine
  '29434',     # medifoxamine
  '6929',      # mianserin
  '588250',    # milnacipran
  '30031',     # minaprine
  '15996',     # mirtazapine
  '31565',     # nefazodone
  '7500',      # nomifensine
  '60842',     # reboxetine
  '258326',    # St. John's wort extract
  '38252',     # tianeptine
  '10737',     # trazodone
  '10898',     # tryptophan
  '39786',     # venlafaxine
  '1086769',   # vilazodone
  '11196',     # viloxazine
  '1455099'    # vortioxetine
]
# '1433212',  # levomilnacipran (no ATC in RxNav)

antidepressant_codes = antidepressant_N06AA_codes + antidepressant_N06AB_codes + antidepressant_N06AF_codes + antidepressant_N06AG_codes + antidepressant_N06AX_codes


# Vasopressors
vasopressor_codes = [
  '3616',   # dobutamine
  '3628',   # dopamine
  '3966',   # ephedrine
  '3992',   # epinephrine
  '6963',   # midodrine
  '7512',   # norepinephrine
  '8163',   # phenylephrine
  '11149'   # vasopressin (USP)
]


medications_cc_registry = {
  # A02A: Antacids
  'antacid': antacid_codes,
  'antacid_A02AA': antacid_A02AA_codes,  # Magnesium compounds
  'antacid_A02AB': antacid_A02AB_codes,  # Aluminium compounds
  'antacid_A02AC': antacid_A02AC_codes,  # Calcium compounds
  
  # A02B: Drugs for peptic ulcer and gastroesophageal reflux disease (GERD)
  'peptic_ulcer_gerd': peptic_ulcer_gerd_codes,
  'peptic_ulcer_gerd_A02BA': peptic_ulcer_gerd_A02BA_codes,  # H2-receptor antagonists
  'peptic_ulcer_gerd_A02BC': peptic_ulcer_gerd_A02BC_codes,  # Proton pump inhibitors
  'peptic_ulcer_gerd_A02BX': peptic_ulcer_gerd_A02BX_codes,  # Other drugs for peptic ulcer and (GERD)
  
  # C01CA: Adrenergic and dopaminergic agents
  'adrenergic_dopaminergic': adrenergic_dopaminergic_codes,
  
  # Anticoagulants
  'anticoagulants': anticoagulant_codes,
  
  # C02: Antihypertensives
  'antihypertensive': antihypertensive_codes,
  'antihypertensive_C02A': antihypertensive_C02A_codes,  # Antiadrenergic agents, centrally acting
  'antihypertensive_C02B': antihypertensive_C02B_codes,  # Antiadrenergic agents, ganglion-blocking
  'antihypertensive_C02C': antihypertensive_C02C_codes,  # Antiadrenergic agents, peripherally acting
  'antihypertensive_C02D': antihypertensive_C02D_codes,  # Agents acting on arteriolar smooth muscle
  'antihypertensive_C02K': antihypertensive_C02K_codes,  # Other antihypertensives
  
  # C07A: Beta blocking agents
  'beta_blocker': beta_blocker_codes,
  
  # C08: Calcium channel blockers
  'calcium_channel_blocker': calcium_channel_blocker_codes,
  'calcium_channel_blocker_C08C': calcium_channel_blocker_C08C_codes,  # Selective calcium channel blockers with mainly vascular effects
  'calcium_channel_blocker_C08D': calcium_channel_blocker_C08D_codes,  # Selective calcium channel blockers with direct cardiac effects
  'calcium_channel_blocker_C08E': calcium_channel_blocker_C08E_codes,  # Non-selective calcium channel blockers
  
  # C09: Agents acting on the renin-angiotensin system
  'renin_angiotensin': renin_angiotensin_codes,
  'renin_angiotensin_C09A': renin_angiotensin_C09A_codes,  # ACE inhibitors, plain
  'renin_angiotensin_C09C': renin_angiotensin_C09C_codes,  # Angiotensin II receptor blockers (ARBs)
  'renin_angiotensin_C09X': renin_angiotensin_C09X_codes,  # Other agents acting on the renin-angiotensin system
  
  # N06: Antidepressants
  'antidepressant': antidepressant_codes,
  'antidepressant_N06AA': antidepressant_N06AA_codes,  # Non-selective monoamine reuptake inhibitors
  'antidepressant_N06AB': antidepressant_N06AB_codes,  # Selective serotonin reuptake inhibitors
  'antidepressant_N06AF': antidepressant_N06AF_codes,  # Monoamine oxidase inhibitors, non-selective
  'antidepressant_N06AG': antidepressant_N06AG_codes,  # Monoamine oxidase A inhibitors
  'antidepressant_N06AX': antidepressant_N06AX_codes,  # Other antidepressants
    
  'vasopressor': vasopressor_codes
}

## gs_pca_kbest_rf  

pipeline: ['f_scaler_1', 'dim_reduc', 'f_scaler_2', 'f_kbest', 'rf_clf']

####  Option 1      

parameters: {
  'dim_reduc__n_components': [None, 20, 15],
  'f_kbest__k': [14, 9, 7],
  'f_kbest__score_func': [<function chi2 at 0x108a7abf8>],
  'rf_clf__criterion': ['gini'],
  'rf_clf__max_depth': [10, 15, 20],
  'rf_clf__n_estimators': [5, 10, 15]
}

Best score: 0.314

Best parameters set:
  dim_reduc__n_components: 20
  f_kbest__k: 7
  f_kbest__score_func: <function chi2 at 0x108a7abf8>
  rf_clf__criterion: 'gini'
  rf_clf__max_depth: 10
  rf_clf__n_estimators: 5

Precision: 1.0
Recall: 0.8
F1 Score: 0.888888888889

Feature Ranking:
  feature no. 1: other (0.25657832345323267)
  feature no. 2: deferral_payments (0.24191669636447494)
  feature no. 3: total_payments (0.128595092401951)
  feature no. 4: exercised_stock_options (0.11711177258717469)
  feature no. 5: bonus (0.11084087045516852)
  feature no. 6: loan_advances (0.08755208887176714)
  feature no. 7: salary (0.05740515586623096)

Accuracy: 0.83573	Precision: 0.33140	Recall: 0.22800	F1: 0.27014	F2: 0.24317
Total predictions: 15000	True positives:  456	False positives:  920	False negatives: 1544	True negatives: 12080



#### Option 2

parameters: {
  'dim_reduc__n_components': [20, 15, 10],
   'f_kbest__k': [7, 5, 3],
   'f_kbest__score_func': [<function f_classif at 0x108a7dae8>,
                           <function chi2 at 0x108a7dbf8>],
   'rf_clf__criterion': ['gini', 'entropy'],
   'rf_clf__max_depth': [20, 30],
   'rf_clf__n_estimators': [2, 5]
 }

Best score: 0.314

Best parameters set:
  dim_reduc__n_components: 20
  f_kbest__k: 7
  f_kbest__score_func: <function chi2 at 0x108a7dbf8>
  rf_clf__criterion: 'gini'
  rf_clf__max_depth: 20
  rf_clf__n_estimators: 5


Precision: 1.0
Recall: 0.8
F1 Score: 0.888888888889

Feature Ranking:
	feature no. 1: other (0.25657832345323267)
	feature no. 2: deferral_payments (0.24191669636447494)
	feature no. 3: total_payments (0.128595092401951)
	feature no. 4: exercised_stock_options (0.11711177258717469)
	feature no. 5: bonus (0.11084087045516852)
	feature no. 6: loan_advances (0.08755208887176714)
	feature no. 7: salary (0.05740515586623096)

Accuracy: 0.83573	Precision: 0.33140	Recall: 0.22800	F1: 0.27014	F2: 0.24317
Total predictions: 15000	True positives:  456	False positives:  920	False negatives: 1544	True negatives: 12080



####  Option 3   

parameters: {
  'dim_reduc__n_components': [None, 22],
  'f_kbest__k': [20, 15, 7, 5],
  'f_kbest__score_func': [<function f_classif at 0x108b84ae8>,
                         <function chi2 at 0x108b84bf8>],
  'rf_clf__criterion': ['gini', 'entropy'],
  'rf_clf__max_depth': [10, 15, 20],
  'rf_clf__n_estimators': [2, 5, 8]
}

Best score: 0.277

Best parameters set:
  dim_reduc__n_components: None
  f_kbest__k: 5
  f_kbest__score_func: <function chi2 at 0x108b84bf8>
  rf_clf__criterion: 'gini'
  rf_clf__max_depth: 10
  rf_clf__n_estimators: 5

Precision: 1.0
Recall: 1.0
F1 Score: 1.0

Feature Ranking:
	feature no. 1: deferral_payments (0.2515986005620595)
	feature no. 2: salary (0.23773341502534118)
	feature no. 3: other (0.22907660995120055)
	feature no. 4: total_payments (0.18288929144995242)
	feature no. 5: exercised_stock_options (0.09870208301144633)

Accuracy: 0.83667	Precision: 0.34043	Recall: 0.24000	F1: 0.28152	F2: 0.25505
Total predictions: 15000	True positives:  480	False positives:  930	False negatives: 1520	True negatives: 12070



### Option 4    

parameters: {
  'dim_reduc__n_components': [20, 15, 10],
  'f_kbest__k': [7, 5, 3],
  'f_kbest__score_func': [<function f_classif at 0x108a7eae8>,
                       <function chi2 at 0x108a7ebf8>],
  'rf_clf__criterion': ['gini', 'entropy'],
  'rf_clf__max_depth': [20, 30],
  'rf_clf__n_estimators': [2, 5]
}

Best score: 0.260

Best parameters set:
  dim_reduc__n_components: 15
  f_kbest__k: 3
  f_kbest__score_func: <function f_classif at 0x108a7eae8>
  rf_clf__criterion: 'gini'
  rf_clf__max_depth: 20
  rf_clf__n_estimators: 5

Precision: 1.0
Recall: 0.8
F1 Score: 0.888888888889

Feature Ranking:
	feature no. 1: other (0.4670890371840632)
	feature no. 2: deferral_payments (0.3018011880778173)
	feature no. 3: salary (0.23110977473811944)

**Accuracy: 0.84593	Precision: 0.38246	Recall: 0.25300	F1: 0.30454	F2: 0.27137
Total predictions: 15000	True positives:  506	False positives:  817	False negatives: 1494	True negatives: 12183**



********************************************************************************



## gs_funion_lnr_sv    

pipeline: ['f_scaler', 'f_union', 'lnr_sv_clf']

#### Option 1

parameters: {
  'f_union__kbest__k': [17, 15, 12],
  'f_union__kbest__score_func': [<function f_classif at 0x108a7aae8>,
                                <function chi2 at 0x108a7abf8>],
  'f_union__pca_scaler__pca__n_components': [None, 20, 15, 10],
  'lnr_sv_clf__C': [1]
}

Best score: 0.258

Best parameters set:
	f_union__kbest__k: 17
	f_union__kbest__score_func: <function f_classif at 0x108a7aae8>
	f_union__pca_scaler__pca__n_components: None
	lnr_sv_clf__C: 1

Precision: 0.75
Recall: 0.6
F1 Score: 0.666666666667

kbest.get_params(): {
  'k': 17,
  'score_func': <function f_classif at 0x108a7aae8>
}

Feature Scores and PValues:
	feature salary : ( score: 18.289684043404513, pval: 3.4782737683651706e-05 )
	feature deferral_payments : ( score: 0.2246112747360051, pval: 0.636281647458697 )
	feature total_payments : ( score: 8.772777730091681, pval: 0.0035893261725152385 )
	feature loan_advances : ( score: 7.184055658288725, pval: 0.008231850906845166 )
	feature bonus : ( score: 20.792252047181538, pval: 1.10129873239521e-05 )
	feature restricted_stock_deferred : ( score: 0.06549965290989124, pval: 0.7983786565630797 )
	feature deferred_income : ( score: 11.458476579280697, pval: 0.0009220367084670714 )
	feature total_stock_value : ( score: 24.182898678566872, pval: 2.4043152760437106e-06 )
	feature expenses : ( score: 6.094173310638967, pval: 0.01475819996537172 )
	feature exercised_stock_options : ( score: 24.815079733218194, pval: 1.8182048777865317e-06 )
	feature other : ( score: 4.1874775069953785, pval: 0.042581747012345836 )
	feature long_term_incentive : ( score: 9.922186013189839, pval: 0.001994181245353672 )
	feature restricted_stock : ( score: 9.212810621977086, pval: 0.002862802957909168 )
	feature director_fees : ( score: 2.126327802007705, pval: 0.14701111317392224 )
	feature to_messages : ( score: 1.6463411294420094, pval: 0.20156265029435688 )
	feature from_poi_to_this_person : ( score: 5.243449713374957, pval: 0.023513867086669714 )
	feature from_messages : ( score: 0.16970094762175436, pval: 0.6810033004207017 )
	feature from_this_person_to_poi : ( score: 2.3826121082276743, pval: 0.12493365956927895 )
	feature shared_receipt_with_poi : ( score: 8.589420731682377, pval: 0.003945802616532257 )
	feature ratio_from_poi : ( score: 3.128091748156737, pval: 0.07911610566379423 )
	feature ratio_to_poi : ( score: 16.40971254803579, pval: 8.388953356704216e-05 )
	feature ratio_tot_stock_value_tot_payments : ( score: 0.022818611212297994, pval: 0.8801456244723183 )
	feature ratio_exercised_stock_tot_stock_value : ( score: 0.04211676849814732, pval: 0.8376933284864809 )


kbest selected features_selected_funion_lnr_sv:
['salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'ratio_from_poi', 'ratio_to_poi']

Accuracy: 0.81707	Precision: 0.29826	Recall: 0.27500	F1: 0.28616	F2: 0.27936
Total predictions: 15000	True positives:  550	False positives: 1294	False negatives: 1450	True negatives: 11706



#### Option 2    

parameters: {
  'f_union__kbest__k': [20, 15, 10],
  'f_union__kbest__score_func': [<function f_classif at 0x108a7eae8>,
                                <function chi2 at 0x108a7ebf8>],
  'f_union__pca_scaler__pca__n_components': [20, 15, 10],
  'lnr_sv_clf__C': [0.001, 0.01, 0.1, 1]
}

Best score: 0.196

Best parameters set:
  f_union__kbest__k: 15
  f_union__kbest__score_func: <function f_classif at 0x108a7eae8>
  f_union__pca_scaler__pca__n_components: 20
  lnr_sv_clf__C: 1

Precision: 1.0
Recall: 0.6
F1 Score: 0.75

kbest.get_params(): {
  'k': 15,
  'score_func': <function f_classif at 0x108a7eae8>
}

Feature Scores and PValues:
	feature salary : ( score: 18.289684043404513, pval: 3.4782737683651706e-05 )
	feature deferral_payments : ( score: 0.2246112747360051, pval: 0.636281647458697 )
	feature total_payments : ( score: 8.772777730091681, pval: 0.0035893261725152385 )
	feature loan_advances : ( score: 7.184055658288725, pval: 0.008231850906845166 )
	feature bonus : ( score: 20.792252047181538, pval: 1.10129873239521e-05 )
	feature restricted_stock_deferred : ( score: 0.06549965290989124, pval: 0.7983786565630797 )
	feature deferred_income : ( score: 11.458476579280697, pval: 0.0009220367084670714 )
	feature total_stock_value : ( score: 24.182898678566872, pval: 2.4043152760437106e-06 )
	feature expenses : ( score: 6.094173310638967, pval: 0.01475819996537172 )
	feature exercised_stock_options : ( score: 24.815079733218194, pval: 1.8182048777865317e-06 )
	feature other : ( score: 4.1874775069953785, pval: 0.042581747012345836 )
	feature long_term_incentive : ( score: 9.922186013189839, pval: 0.001994181245353672 )
	feature restricted_stock : ( score: 9.212810621977086, pval: 0.002862802957909168 )
	feature director_fees : ( score: 2.126327802007705, pval: 0.14701111317392224 )
	feature to_messages : ( score: 1.6463411294420094, pval: 0.20156265029435688 )
	feature from_poi_to_this_person : ( score: 5.243449713374957, pval: 0.023513867086669714 )
	feature from_messages : ( score: 0.16970094762175436, pval: 0.6810033004207017 )
	feature from_this_person_to_poi : ( score: 2.3826121082276743, pval: 0.12493365956927895 )
	feature shared_receipt_with_poi : ( score: 8.589420731682377, pval: 0.003945802616532257 )
	feature ratio_from_poi : ( score: 3.128091748156737, pval: 0.07911610566379423 )
	feature ratio_to_poi : ( score: 16.40971254803579, pval: 8.388953356704216e-05 )
	feature ratio_tot_stock_value_tot_payments : ( score: 0.022818611212297994, pval: 0.8801456244723183 )
	feature ratio_exercised_stock_tot_stock_value : ( score: 0.04211676849814732, pval: 0.8376933284864809 )

kbest selected features_selected_funion_lnr_sv:
['salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'ratio_from_poi', 'ratio_to_poi']

Accuracy: 0.83267	Precision: 0.30564	Recall: 0.20050	F1: 0.24215	F2: 0.21531
Total predictions: 15000	True positives:  401	False positives:  911	False negatives: 1599	True negatives: 12089



**#### Option 3 !!!!! THIS IS BEST SOLUTION SO FAR**

parameters: {
  'f_union__kbest__k': [20, 17],
  'f_union__kbest__score_func': [<function f_classif at 0x108b84ae8>,
                              <function chi2 at 0x108b84bf8>],
  'f_union__pca_scaler__pca__n_components': [None, 22],
  'lnr_sv_clf__C': [1, 10]}

Best score: 0.390

Best parameters set:
	f_union__kbest__k: 17
	f_union__kbest__score_func: <function f_classif at 0x108b84ae8>
	f_union__pca_scaler__pca__n_components: 22
	lnr_sv_clf__C: 10

Precision: 0.666666666667
Recall: 0.8
F1 Score: 0.727272727273

kbest.get_params():
  {'k': 17, 'score_func': <function f_classif at 0x108b84ae8>}

Feature Scores and PValues:
	feature salary : ( score: 18.289684043404513, pval: 3.4782737683651706e-05 )
	feature deferral_payments : ( score: 0.2246112747360051, pval: 0.636281647458697 )
	feature total_payments : ( score: 8.772777730091681, pval: 0.0035893261725152385 )
	feature loan_advances : ( score: 7.184055658288725, pval: 0.008231850906845166 )
	feature bonus : ( score: 20.792252047181538, pval: 1.10129873239521e-05 )
	feature restricted_stock_deferred : ( score: 0.06549965290989124, pval: 0.7983786565630797 )
	feature deferred_income : ( score: 11.458476579280697, pval: 0.0009220367084670714 )
	feature total_stock_value : ( score: 24.182898678566872, pval: 2.4043152760437106e-06 )
	feature expenses : ( score: 6.094173310638967, pval: 0.01475819996537172 )
	feature exercised_stock_options : ( score: 24.815079733218194, pval: 1.8182048777865317e-06 )
	feature other : ( score: 4.1874775069953785, pval: 0.042581747012345836 )
	feature long_term_incentive : ( score: 9.922186013189839, pval: 0.001994181245353672 )
	feature restricted_stock : ( score: 9.212810621977086, pval: 0.002862802957909168 )
	feature director_fees : ( score: 2.126327802007705, pval: 0.14701111317392224 )
	feature to_messages : ( score: 1.6463411294420094, pval: 0.20156265029435688 )
	feature from_poi_to_this_person : ( score: 5.243449713374957, pval: 0.023513867086669714 )
	feature from_messages : ( score: 0.16970094762175436, pval: 0.6810033004207017 )
	feature from_this_person_to_poi : ( score: 2.3826121082276743, pval: 0.12493365956927895 )
	feature shared_receipt_with_poi : ( score: 8.589420731682377, pval: 0.003945802616532257 )
	feature ratio_from_poi : ( score: 3.128091748156737, pval: 0.07911610566379423 )
	feature ratio_to_poi : ( score: 16.40971254803579, pval: 8.388953356704216e-05 )
	feature ratio_tot_stock_value_tot_payments : ( score: 0.022818611212297994, pval: 0.8801456244723183 )
	feature ratio_exercised_stock_tot_stock_value : ( score: 0.04211676849814732, pval: 0.8376933284864809 )

kbest selected features_selected_funion_lnr_sv:
['salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'ratio_from_poi', 'ratio_to_poi']

**Accuracy: 0.84007	Precision: 0.39878	Recall: 0.39300	F1: 0.39587	F2: 0.39414
Total predictions: 15000	True positives:  786	False positives: 1185	False negatives: 1214	True negatives: 11815**



#### Option 4:  

parameters: {
  'f_union__kbest__k': [20, 15, 10],
  'f_union__kbest__score_func': [<function f_classif at 0x108a7dae8>,
                                <function chi2 at 0x108a7dbf8>],
  'f_union__pca_scaler__pca__n_components': [20, 15, 10],
  'lnr_sv_clf__C': [0.001, 0.01, 0.1, 1]}

Best score: 0.200

Best parameters set:
	f_union__kbest__k: 15
	f_union__kbest__score_func: <function f_classif at 0x108a7dae8>
	f_union__pca_scaler__pca__n_components: 20
	lnr_sv_clf__C: 1

Precision: 1.0
Recall: 0.6
F1 Score: 0.75

kbest.get_params():
  {'k': 15, 'score_func': <function f_classif at 0x108a7dae8>}

Feature Scores and PValues:
	feature salary : ( score: 18.289684043404513, pval: 3.4782737683651706e-05 )
	feature deferral_payments : ( score: 0.2246112747360051, pval: 0.636281647458697 )
	feature total_payments : ( score: 8.772777730091681, pval: 0.0035893261725152385 )
	feature loan_advances : ( score: 7.184055658288725, pval: 0.008231850906845166 )
	feature bonus : ( score: 20.792252047181538, pval: 1.10129873239521e-05 )
	feature restricted_stock_deferred : ( score: 0.06549965290989124, pval: 0.7983786565630797 )
	feature deferred_income : ( score: 11.458476579280697, pval: 0.0009220367084670714 )
	feature total_stock_value : ( score: 24.182898678566872, pval: 2.4043152760437106e-06 )
	feature expenses : ( score: 6.094173310638967, pval: 0.01475819996537172 )
	feature exercised_stock_options : ( score: 24.815079733218194, pval: 1.8182048777865317e-06 )
	feature other : ( score: 4.1874775069953785, pval: 0.042581747012345836 )
	feature long_term_incentive : ( score: 9.922186013189839, pval: 0.001994181245353672 )
	feature restricted_stock : ( score: 9.212810621977086, pval: 0.002862802957909168 )
	feature director_fees : ( score: 2.126327802007705, pval: 0.14701111317392224 )
	feature to_messages : ( score: 1.6463411294420094, pval: 0.20156265029435688 )
	feature from_poi_to_this_person : ( score: 5.243449713374957, pval: 0.023513867086669714 )
	feature from_messages : ( score: 0.16970094762175436, pval: 0.6810033004207017 )
	feature from_this_person_to_poi : ( score: 2.3826121082276743, pval: 0.12493365956927895 )
	feature shared_receipt_with_poi : ( score: 8.589420731682377, pval: 0.003945802616532257 )
	feature ratio_from_poi : ( score: 3.128091748156737, pval: 0.07911610566379423 )
	feature ratio_to_poi : ( score: 16.40971254803579, pval: 8.388953356704216e-05 )
	feature ratio_tot_stock_value_tot_payments : ( score: 0.022818611212297994, pval: 0.8801456244723183 )
	feature ratio_exercised_stock_tot_stock_value : ( score: 0.04211676849814732, pval: 0.8376933284864809 )

kbest selected features_selected_funion_lnr_sv:
  ['salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'ratio_from_poi', 'ratio_to_poi']

Accuracy: 0.83267	Precision: 0.30564	Recall: 0.20050	F1: 0.24215	F2: 0.21531
Total predictions: 15000	True positives:  401	False positives:  911	False negatives: 1599	True negatives: 12089



********************************************************************************



## gs_kbest_pca_rf    

pipeline: ['f_scaler_1', 'f_kbest', 'dim_reduc', 'rf_clf']

#### Option 1    

parameters: {
  'dim_reduc__n_components': [7, 5, 3],
  'f_kbest__k': [20, 15, 10],
  'f_kbest__score_func': [<function f_classif at 0x108a7eae8>,
                         <function chi2 at 0x108a7ebf8>],
  'rf_clf__criterion': ['gini', 'entropy'],
  'rf_clf__max_depth': [20, 30],
  'rf_clf__n_estimators': [5, 12]
}

 Best score: 0.195

 Best parameters set:
  dim_reduc__n_components: 7
  f_kbest__k: 10
  f_kbest__score_func: <function chi2 at 0x108a7ebf8>
  rf_clf__criterion: 'entropy'
  rf_clf__max_depth: 20
  rf_clf__n_estimators: 5

 Precision: 1.0
 Recall: 0.8
 F1 Score: 0.888888888889

Feature Ranking:
  feature no. 1: total_payments (0.25414277069130786)
  feature no. 2: exercised_stock_options (0.17055624511678902)
  feature no. 3: total_stock_value (0.16050466797376114)
  feature no. 4: salary (0.12507110406912553)
  feature no. 5: bonus (0.11768662895034061)
  feature no. 6: loan_advances (0.1168323427706521)
  feature no. 7: other (0.0552062404280237)

Accuracy: 0.83533	Precision: 0.30514	Recall: 0.18400	F1: 0.22957	F2: 0.19987
Total predictions: 15000	True positives:  368	False positives:  838	False negatives: 1632	True negatives: 12162



#### Option 2

parameters: {
  'dim_reduc__n_components': [7, 5, 3],
  'f_kbest__k': [20, 15, 10],
  'f_kbest__score_func': [<function f_classif at 0x108a7dae8>,
                         <function chi2 at 0x108a7dbf8>],
  'rf_clf__criterion': ['gini', 'entropy'],
  'rf_clf__max_depth': [20, 30],
  'rf_clf__n_estimators': [5, 12]
}

Best score: 0.260

Best parameters set:
  dim_reduc__n_components: 5
  f_kbest__k: 20
  f_kbest__score_func: <function chi2 at 0x108a7dbf8>
  rf_clf__criterion: 'entropy'
  rf_clf__max_depth: 20
  rf_clf__n_estimators: 5

Precision: 1.0
Recall: 0.8
F1 Score: 0.888888888889

Feature Ranking:
	feature no. 1: salary (0.3723075690037799)
	feature no. 2: deferral_payments (0.23866102552083235)
	feature no. 3: bonus (0.17434441095328976)
	feature no. 4: total_payments (0.10805908541163647)
	feature no. 5: loan_advances (0.10662790911046145)

Accuracy: 0.82360	Precision: 0.24077	Recall: 0.15000	F1: 0.18484	F2: 0.16223
Total predictions: 15000	True positives:  300	False positives:  946	False negatives: 1700	True negatives: 12054



********************************************************************************



### pca_kbest_lnr_sv     

pipeline: ['f_scaler_1', 'dim_reduc', 'f_scaler_2', 'f_kbest', 'lnr_sv_clf']

#### Option 1   WORST EVER
parameters: {
  'dim_reduc__n_components': [20, 15, 10],
  'f_kbest__k': [7, 5, 3],
  'f_kbest__score_func': [<function f_classif at 0x108a7eae8>,
                         <function chi2 at 0x108a7ebf8>],
  'lnr_sv_clf__C': [0.001, 0.01, 0.1, 1]}

Best score: 0.017

Best parameters set:
  dim_reduc__n_components: 20
  f_kbest__k: 7
  f_kbest__score_func: <function f_classif at 0x108a7eae8>
  lnr_sv_clf__C: 1

Precision: 0.0
Recall: 0.0
F1 Score: 0.0

kbest.scores_:
[ 10.65171629  14.45462963   4.84145622   1.12564647   1.42772788
  1.03906525   0.03264966   1.59449204   1.22357837   4.19453053
 12.08634607   0.14008301   0.28218339   1.16532697   0.08374354
  0.34293519   0.01854338   0.6203272    0.38597786   0.90302865]
kbest.pvalues_:
[  1.37992623e-03   2.13038163e-04   2.94110768e-02   2.90519801e-01
  2.34141587e-01   3.09784801e-01   8.56868666e-01   2.08769827e-01
  2.70543261e-01   4.24102746e-02   6.75532238e-04   7.08760764e-01
  5.96109512e-01   2.82206630e-01   7.72712029e-01   5.59075983e-01
  8.91877884e-01   4.32247718e-01   5.35424326e-01   3.43596923e-01]

kbest.get_params():
  {'k': 7, 'score_func': <function f_classif at 0x108a7eae8>}

features_selected_pca_kbest_lnr_sv:
  ['salary', 'deferral_payments', 'total_payments', 'bonus', 'total_stock_value', 'exercised_stock_options', 'other']

Accuracy: 0.85553	Precision: 0.13216	Recall: 0.01500	F1: 0.02694	F2: 0.01823
Total predictions: 15000	True positives:   30	False positives:  197	False negatives: 1970	True negatives: 12803



#### Option 2   

parameters: {
  'dim_reduc__n_components': [20, 15, 10],
  'f_kbest__k': [7, 5, 3],
  'f_kbest__score_func': [<function f_classif at 0x108a7dae8>,
                         <function chi2 at 0x108a7dbf8>],
  'lnr_sv_clf__C': [0.001, 0.01, 0.1, 1]
}

Best score: 0.023

Best parameters set:
  dim_reduc__n_components: 20
  f_kbest__k: 7
  f_kbest__score_func: <function f_classif at 0x108a7dae8>
  lnr_sv_clf__C: 1

Precision: 0.0
Recall: 0.0
F1 Score: 0.0

kbest.get_params():
  {'k': 7, 'score_func': <function f_classif at 0x108a7dae8>}

features_selected_pca_kbest_lnr_sv:
  ['salary', 'deferral_payments', 'total_payments', 'bonus', 'total_stock_value', 'exercised_stock_options', 'other']

Accuracy: 0.85553	Precision: 0.13216	Recall: 0.01500	F1: 0.02694	F2: 0.01823
Total predictions: 15000	True positives:   30	False positives:  197	False negatives: 1970	True negatives: 12803

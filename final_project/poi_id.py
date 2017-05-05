

import pickle
import numpy
import matplotlib.pyplot
import math
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


with open("final_project_dataset.pkl", "rb") as f:
    enron_data = pickle.load(f)


#########################################################################
#
#  Data Exploration
#
#########################################################################


### Counts of total data point and features
tot_data_points = len(enron_data)
tot_features = len(enron_data["SKILLING JEFFREY K"])


# print("Number of Data Points: ", str(tot_data_points))
# print("Number of Features: ", str(tot_features))


### Feature Lists by Data Types
numeric_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
bool_features = ['poi']
text_features = ['email_address']
orig_features = numeric_features + text_features


### Function to count quantifiable values for feature, optional setting POI
def get_quantifiable_count(key, is_poi = None):
    count = 0
    for k, v in enron_data.items():
        if is_poi is not None:
            if v[key] != 'NaN' and v['poi'] == is_poi:
                count += 1
        else:
            if v[key] != 'NaN':
                count += 1
    return count

### Count of POI and non-POI in data set
int_poi_count = get_quantifiable_count('poi', True)
int_non_poi_count = get_quantifiable_count('poi', False)


# print("Number of POIs: ", str(int_poi_count))
# print("Number of non-POIs: ", str(int_non_poi_count))


quantifiable_poi = get_quantifiable_count('poi')
quantifiable_email = get_quantifiable_count('email_address')

### No missing data for POI feature - good
# print("Non-NaN 'poi'", str(quantifiable_poi))
# print("Percentage of Non-NaN 'poi'", str(float(quantifiable_poi)/float(tot_data_points)))

### Some missing data for email_address feature
# print("Non-NaN 'email_address'", str(quantifiable_email))
# print("Percentage of Non-NaN 'email_address'", str(float(quantifiable_email)/float(tot_data_points)))


### Build numpy array of count and density of non-NaN / quantifiable feature values for numeric features
orig_feature_quantifiable_count = []
orig_feature_quantifiable_density = []
orig_feature_poi_quantifiable_count = []
orig_feature_poi_quantifiable_density = []
orig_feature_non_poi_quantifiable_count = []
orig_feature_non_poi_quantifiable_density = []

for f in  orig_features:
    orig_feature_quantifiable_count.append(get_quantifiable_count(f))
    orig_feature_quantifiable_density.append(round(float(get_quantifiable_count(f))/float(tot_data_points), 5))
    orig_feature_poi_quantifiable_count.append(get_quantifiable_count(f, True))
    orig_feature_poi_quantifiable_density.append(round(float(get_quantifiable_count(f, True))/float(int_poi_count), 4))
    orig_feature_non_poi_quantifiable_count.append(get_quantifiable_count(f, False))
    orig_feature_non_poi_quantifiable_density.append(round(float(get_quantifiable_count(f, False))/float(int_non_poi_count), 4))

orig_feature_count_density = numpy.array([
[feature, count, density, poi_count, poi_density, non_poi_count, non_poi_density]
for feature, count, density, poi_count, poi_density, non_poi_count, non_poi_density in zip(orig_features, orig_feature_quantifiable_count, orig_feature_quantifiable_density, orig_feature_poi_quantifiable_count, orig_feature_poi_quantifiable_density, orig_feature_non_poi_quantifiable_count, orig_feature_non_poi_quantifiable_density)])
# print(orig_feature_count_density)


### Build sorted dictionary with highest number of missing data of features for each data point

NaN_feature_count = {}
for k, v in enron_data.items():
    count = 0
    for subK, subV in v.items():
        if subV == 'NaN':
            count += 1
    if count > 16: ## if more than 16 missing examine the data point
        NaN_feature_count[k] = count
# print(NaN_feature_count)

### Sort dictionary NaN_feature_count
sorted_NaN_feature_count = [(k, NaN_feature_count[k]) for k in sorted(NaN_feature_count, key=NaN_feature_count.get, reverse=True)]

# print("%s got the highest count of missing data at %s" % (max(NaN_feature_count, key=NaN_feature_count.get), str(max(NaN_feature_count.values()))))
# print(sorted_NaN_feature_count)


#########################################################################
#
#  Examine and Remove Outliers by Visulization
#
#########################################################################
### EDA Plotting function
def exploration_plot(dataset, x_feature, y_feature):
    explore_plot_features = [x_feature, y_feature]
    plt_data = featureFormat(dataset, explore_plot_features)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)

    for point in plt_data:
        total_stock_value = point[0]
        total_payments = point[1]
        # print(type(point))
        matplotlib.pyplot.scatter( total_stock_value, total_payments )
        ax.annotate('(%s, %s)' % (point[1], point[0]), xy=point, textcoords='data') # <--

    matplotlib.pyplot.xlabel(str(x_feature))
    matplotlib.pyplot.ylabel(str(y_feature))
    matplotlib.pyplot.show()

# exploration_plot(enron_data, 'total_stock_value', 'total_payments')

### Task 2: Remove outliers
enron_data.pop("TOTAL", 0) # Remove outlier predetermined from mini-project
enron_data.pop('LOCKHART EUGENE E', 0) # has no data
enron_data.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Obviously travel agency has nothing to do with Enron

# exploration_plot(enron_data, 'salary', 'other')
# exploration_plot(enron_data, 'salary', 'bonus')
# exploration_plot(enron_data, 'salary', 'expenses')
# exploration_plot(enron_data, 'salary', 'exercised_stock_options')


#########################################################################
#
#  New Feature Creations
#
#########################################################################

#### New Features I would like to add: ratio of total_stock_value to total_payments, ratio of exercised_stock_options to total_stock_value, ratio of from_poi_to_this_person to to_messages, ratio of from_this_person_to_poi to from_messages

### Function to Compute Ratios
def computeRatio( numerator, denominator ):

    ### in case of numerator or denominator having "NaN" value, return 0.
    ratio = float(numerator)/float(denominator)
    ratio = ratio if not math.isnan(ratio) else 0
    return ratio


new_features = ["ratio_from_poi", "ratio_to_poi", "ratio_tot_stock_value_tot_payments", "ratio_exercised_stock_tot_stock_value"]

for name in enron_data:

    data_point = enron_data[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]

    total_stock_value = data_point["total_stock_value"]
    total_payments = data_point["total_payments"]
    exercised_stock_options = data_point["exercised_stock_options"]

    ratio_from_poi = computeRatio( from_poi_to_this_person, to_messages )
    ratio_to_poi = computeRatio( from_this_person_to_poi, from_messages )
    ratio_tot_stock_value_tot_payments = computeRatio( total_stock_value, total_payments )
    ratio_exercised_stock_tot_stock_value = computeRatio( exercised_stock_options, total_stock_value )

    data_point["ratio_from_poi"] = ratio_from_poi
    data_point["ratio_to_poi"] = ratio_to_poi
    data_point["ratio_tot_stock_value_tot_payments"] = ratio_tot_stock_value_tot_payments
    data_point["ratio_exercised_stock_tot_stock_value"] = ratio_exercised_stock_tot_stock_value

new_feature_quantifiable_count = []
new_feature_quantifiable_density = []
new_feature_poi_quantifiable_count = []
new_feature_poi_quantifiable_density = []
new_feature_non_poi_quantifiable_count = []
new_feature_non_poi_quantifiable_density = []

for f in  new_features:
    new_feature_quantifiable_count.append(get_quantifiable_count(f))
    new_feature_quantifiable_density.append(round(float(get_quantifiable_count(f))/float(tot_data_points), 5))
    new_feature_poi_quantifiable_count.append(get_quantifiable_count(f, True))
    new_feature_poi_quantifiable_density.append(round(float(get_quantifiable_count(f, True))/float(int_poi_count), 4))
    new_feature_non_poi_quantifiable_count.append(get_quantifiable_count(f, False))
    new_feature_non_poi_quantifiable_density.append(round(float(get_quantifiable_count(f, False))/float(int_non_poi_count), 4))

new_feature_count_density = numpy.array([
[feature, count, density, poi_count, poi_density, non_poi_count, non_poi_density]
for feature, count, density, poi_count, poi_density, non_poi_count, non_poi_density in zip(new_features, new_feature_quantifiable_count, new_feature_quantifiable_density, new_feature_poi_quantifiable_count, new_feature_poi_quantifiable_density, new_feature_non_poi_quantifiable_count, new_feature_non_poi_quantifiable_density)])
# print(new_feature_count_density)

### New All Original and New Feature List
all_features = bool_features + numeric_features + new_features

### Store to my_dataset for easy export below.
my_dataset = enron_data

# print("\t%s: %r" % (param_name, best_parameters[param_name]))
### Extract features and labels from dataset for local testing
# print('all %d features :: %r' % (len(all_features), all_features))
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


#########################################################################
#
#  Estimators and Pipelines
#
#########################################################################


from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from pprint import pprint
from time import time
from tester import dump_classifier_and_data, test_classifier

### Build Estimators for Pipeline
f_minmaxscaler = preprocessing.MinMaxScaler()
f_stdscaler = preprocessing.StandardScaler()
f_kbest = feature_selection.SelectKBest()
dim_reduc = PCA(svd_solver='randomized', random_state=42)
lnr_sv_clf = SVC(kernel="linear", random_state=42)
rbf_sv_clf = SVC(kernel="rbf", random_state=42)
dt_clf = DecisionTreeClassifier(random_state=42)
nb_clf = GaussianNB()
rf_clf = RandomForestClassifier(random_state=42)

pca_scaler_est = [('pca', dim_reduc), ('stdscaler', f_stdscaler)]
pca_scaler = Pipeline(pca_scaler_est)
kbest_scaler_est = [('kbest', f_kbest), ('stdscaler', f_stdscaler)]
kbest_scaler = Pipeline(kbest_scaler_est)
f_union = FeatureUnion([("kbest", f_kbest), ("pca_scaler", pca_scaler)])


### First I tried to perform quick fit and score for different order and combinations of estimators and classifiers with pipeline and feature_union.
estimators = []
pipes = []
def add_estimator_pipe(estimator, pipe):
    estimators.append(estimators)
    pipes.append(pipe)


###0 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, linear svc
est_pipe_pca_kbest_lnr_sv = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('lnr_sv_clf', lnr_sv_clf)]
pipe_pca_kbest_lnr_sv = Pipeline(est_pipe_pca_kbest_lnr_sv)
add_estimator_pipe(est_pipe_pca_kbest_lnr_sv, pipe_pca_kbest_lnr_sv)

###1 Scaler, kbest, pca, linear svc
est_pipe_kbest_pca_lnr_sv = [('f_scaler_1', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('lnr_sv_clf', lnr_sv_clf)]
pipe_kbest_pca_lnr_sv = Pipeline(est_pipe_kbest_pca_lnr_sv)
add_estimator_pipe(est_pipe_kbest_pca_lnr_sv, pipe_kbest_pca_lnr_sv)

###2 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, rbf svc
est_pipe_pca_kbest_rbf_sv = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('rbf_sv_clf', rbf_sv_clf)]
pipe_pca_kbest_rbf_sv = Pipeline(est_pipe_pca_kbest_rbf_sv)
add_estimator_pipe(est_pipe_pca_kbest_rbf_sv, pipe_pca_kbest_rbf_sv)

###3 Scaler, kbest, pca, rbf svc
est_pipe_kbest_pca_rbf_sv = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('rbf_sv_clf', rbf_sv_clf)]
pipe_kbest_pca_rbf_sv = Pipeline(est_pipe_kbest_pca_rbf_sv)
add_estimator_pipe(est_pipe_kbest_pca_rbf_sv, pipe_kbest_pca_rbf_sv)

###4 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, decision tree classifier
est_pipe_pca_kbest_dt = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('dt_clf', dt_clf)]
pipe_pca_kbest_dt = Pipeline(est_pipe_pca_kbest_dt)
add_estimator_pipe(est_pipe_pca_kbest_dt, pipe_pca_kbest_dt)

###5 Scaler, kbest, pca, decision tree classifier
est_pipe_kbest_pca_dt = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('dt_clf', dt_clf)]
pipe_kbest_pca_dt = Pipeline(est_pipe_kbest_pca_dt)
add_estimator_pipe(est_pipe_kbest_pca_dt, pipe_kbest_pca_dt)

###6 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, gaussian naive bayes classifier
est_pipe_pca_kbest_nb = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('nb_clf', nb_clf)]
pipe_pca_kbest_nb = Pipeline(est_pipe_pca_kbest_nb)
add_estimator_pipe(est_pipe_pca_kbest_nb, pipe_pca_kbest_nb)

###7 Scaler, kbest, pca, gaussian naive bayes classifier
est_pipe_kbest_pca_nb = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('nb_clf', nb_clf)]
pipe_kbest_pca_nb = Pipeline(est_pipe_kbest_pca_nb)
add_estimator_pipe(est_pipe_kbest_pca_nb, pipe_kbest_pca_nb)

###8 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, random forest classifier
est_pipe_pca_kbest_rf = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('rf_clf', rf_clf)]
pipe_pca_kbest_rf = Pipeline(est_pipe_pca_kbest_rf)
add_estimator_pipe(est_pipe_pca_kbest_rf, pipe_pca_kbest_rf)

###9 Scaler, kbest, pca, random forest classifier
est_pipe_kbest_pca_rf = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('rf_clf', rf_clf)]
pipe_kbest_pca_rf = Pipeline(est_pipe_kbest_pca_rf)
add_estimator_pipe(est_pipe_kbest_pca_rf, pipe_kbest_pca_rf)

###10 Scaler, featureUnion[pca & kbest], linear svc
est_funion_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('lnr_sv_clf', lnr_sv_clf)]
funion_lnr_sv = Pipeline(est_funion_lnr_sv)
add_estimator_pipe(est_funion_lnr_sv, funion_lnr_sv)

###11 Scaler, featureUnion[pca & kbest], rbf svc
est_funion_rbf_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rbf_sv_clf', rbf_sv_clf)]
funion_rbf_sv = Pipeline(est_funion_rbf_sv)
add_estimator_pipe(est_funion_rbf_sv, funion_rbf_sv)

###12 Scaler, featureUnion[pca & kbest], random forest classifier
est_funion_rf = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rf_clf', rf_clf)]
funion_rf = Pipeline(est_funion_rf)
add_estimator_pipe(est_funion_rf, funion_rf)

# print(len(estimators), " estimators :: ", estimators)
# print(len(pipes), " pipes :: ", pipes)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Quick run with training set to roughly select the model with highest f1, precision and recall scores
i = 0
for estimator, pipe in zip(estimators, pipes):
    # print("\n=================================================================")
    # for name, est in estimator:
    #     print("\nParameters for ", name, " :: \n", est.get_params().keys())


    pipe.fit(features_train, labels_train)
    labels_pred = pipe.predict(features_test)
    clf_reports = classification_report(labels_test, labels_pred)
    # print("\nClassification Report for ", i, "\n", clf_reports)
    i += 1
    # print("\n=================================================================")


################################################################################
# The 3 following gives the best scores (See attached text files):
# 1) #8: minmaxscaler, pca, stdscaler, kbest, random forest classifier (f1: 0.88, recall: 0.91, precision: 0.92)
# 2) #0: minmaxscaler, pca, stdscaler, kbest, linear svc (f1: 0.89, recall: 0.88, precision: 0.89)
# 3) #9: stdscaler, kbest, pca, random forest classifier (f1: 0.88, recall: 0.88, precision: 0.87)
################################################################################


# I continue tweaking the three options above, using minmaxscaler and stdscaler
# interchangably, changing sequance of the scalers, pca and SelectKBest.
# Below are the optimized options, tested on cv
# cv = StratifiedShuffleSplit(n_splits=100, random_state=42, test_size=0.1)

###9 stdscaler, kbest, pca, random forest classifier
est_pipe_kbest_pca_rf = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('rf_clf', rf_clf)]
pipe_kbest_pca_rf = Pipeline(est_pipe_kbest_pca_rf)


###8 minmaxscaler, pca, stdscaler, kbest, random forest classifier
est_pipe_pca_kbest_rf = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_scaler_2', f_minmaxscaler),  ('f_kbest', f_kbest), ('rf_clf', rf_clf)]
pipe_pca_kbest_rf = Pipeline(est_pipe_pca_kbest_rf)


###0 minmaxscaler, pca, stdscaler, kbest, linear svc
est_pipe_pca_kbest_lnr_sv = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_scaler_2', f_minmaxscaler),  ('f_kbest', f_kbest), ('lnr_sv_clf', lnr_sv_clf)]
pipe_pca_kbest_lnr_sv = Pipeline(est_pipe_pca_kbest_lnr_sv)

###10 Scaler, featureUnion[pca & kbest], linear svc
est_funion_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('lnr_sv_clf', lnr_sv_clf)]
funion_lnr_sv = Pipeline(est_funion_lnr_sv)


### Setting up Param_Grids for GridSearchCV ####
params_pipe_kbest_pca_rf = dict(
f_kbest__k = [20, 15, 10],
f_kbest__score_func = [f_classif, chi2],
dim_reduc__n_components = [7, 5, 3],
rf_clf__n_estimators = [5, 12],
rf_clf__criterion = ['gini', 'entropy'],
rf_clf__max_depth = [20, 30]
)
params_pipe_pca_kbest_rf = dict(
dim_reduc__n_components = [None, 22],
f_kbest__k = [20, 15, 7, 5],
f_kbest__score_func = [f_classif, chi2],
rf_clf__n_estimators = [2, 5, 8],
rf_clf__criterion  = ['gini', 'entropy'],
rf_clf__max_depth = [10, 15, 20]
)
params_pca_kbest_lnr_sv = dict(
dim_reduc__n_components = [20, 15, 10],
f_kbest__k = [7, 5, 3],
f_kbest__score_func = [f_classif, chi2],
lnr_sv_clf__C = [0.001, 0.01, 0.1, 1]
)
params_funion_lnr_sv = dict(
f_union__pca_scaler__pca__n_components = [None, 22],
f_union__kbest__k = [20, 17],
f_union__kbest__score_func = [f_classif, chi2],
lnr_sv_clf__C = [1, 10, 100]
)

grid_search_estimators = [est_pipe_kbest_pca_rf, est_pipe_pca_kbest_rf, est_pipe_pca_kbest_lnr_sv]
grid_search_pipes = [pipe_kbest_pca_rf, pipe_pca_kbest_rf, pipe_pca_kbest_lnr_sv]
grid_search_params = [params_pipe_kbest_pca_rf, params_pipe_pca_kbest_rf, params_pca_kbest_lnr_sv]

### Mimic StratifiedShuffleSplit Cross Validator used in tester.py
cv = StratifiedShuffleSplit(n_splits=100, random_state=42)


# print('    #####################################################################    ')
# gs_kbest_pca_rf = GridSearchCV(pipe_kbest_pca_rf, param_grid=params_pipe_kbest_pca_rf, cv=cv, scoring='f1')
#
# print("Performing grid search gs_kbest_pca_rf")
# print("pipeline:", [name for name, _ in pipe_kbest_pca_rf.steps])
# pprint(pipe_kbest_pca_rf)
# print("parameters:")
# pprint(params_pipe_kbest_pca_rf)
# t0 = time()
# gs_kbest_pca_rf.fit(features, labels)
# print("done in %0.3fs" % (time() - t0))
#
# print("Best score: %0.3f" % gs_kbest_pca_rf.best_score_)
# best_estimator_kbest_pca_rf = gs_kbest_pca_rf.best_estimator_
# print("Best estimator pipeline: ", gs_kbest_pca_rf.best_estimator_)
# print("Best parameters set: ")
# best_parameters_kbest_pca_rf = gs_kbest_pca_rf.best_estimator_.get_params()
# for param_name in sorted(params_pipe_kbest_pca_rf.keys()):
#     print("\t%s: %r" % (param_name, best_parameters_kbest_pca_rf[param_name]))
#
# pred_kbest_pca_rf = gs_kbest_pca_rf.predict(features_test)
# print('Precision:', precision_score(labels_test, pred_kbest_pca_rf))
# print('Recall:', recall_score(labels_test, pred_kbest_pca_rf))
# print('F1 Score:', f1_score(labels_test, pred_kbest_pca_rf))
#
#
# ### create a new list that contains the features selected by SelectKBest
# features_selected_kbest_pca_rf = [all_features[i+1] for i in gs_kbest_pca_rf.best_estimator_.named_steps['f_kbest'].get_support(indices=True)]
#
# ### Access the feature importances from Random Forest Classifier
# importances_kbest_pca_rf = gs_kbest_pca_rf.best_estimator_.named_steps['rf_clf'].feature_importances_
# indices = numpy.argsort(importances_kbest_pca_rf)[::-1]
#
# # Use features_selected, the features selected by SelectKBest, and not features_list
# print('Feature Ranking: ')
# for i in range(min(len(features_selected_kbest_pca_rf), len(importances_kbest_pca_rf))):
#     print("\tfeature no. {}: {} ({})".format(i+1,features_selected_kbest_pca_rf[indices[i]],importances_kbest_pca_rf[indices[i]]))
#
# ### Test best_estimator_ with tester.test_classifier
# test_classifier(best_estimator_kbest_pca_rf, my_dataset, all_features)





print('    #####################################################################    ')
gs_pca_kbest_rf = GridSearchCV(pipe_pca_kbest_rf, param_grid=params_pipe_pca_kbest_rf, cv=cv, scoring='f1')

print("Performing grid search gs_pca_kbest_rf")
print("pipeline:", [name for name, _ in pipe_pca_kbest_rf.steps])
print("parameters:")
pprint(params_pipe_pca_kbest_rf)
t0 = time()
gs_pca_kbest_rf.fit(features, labels)
print("done in %0.3fs" % (time() - t0))

print("Best score: %0.3f" % gs_pca_kbest_rf.best_score_)
best_estimator_pca_kbest_rf = gs_pca_kbest_rf.best_estimator_
print("Best estimator pipeline: ", gs_pca_kbest_rf.best_estimator_)
print("Best parameters set: ")
best_parameters_pca_kbest_rf = gs_pca_kbest_rf.best_estimator_.get_params()
for param_name in sorted(params_pipe_pca_kbest_rf.keys()):
    print("\t%s: %r" % (param_name, best_parameters_pca_kbest_rf[param_name]))

pred_pca_kbest_rf = gs_pca_kbest_rf.predict(features_test)
print('Precision:', precision_score(labels_test, pred_pca_kbest_rf))
print('Recall:', recall_score(labels_test, pred_pca_kbest_rf))
print('F1 Score:', f1_score(labels_test, pred_pca_kbest_rf))


### create a new list that contains the features selected by SelectKBest
features_selected_pca_kbest_rf = [all_features[i+1] for i in gs_pca_kbest_rf.best_estimator_.named_steps['f_kbest'].get_support(indices=True)]

### Access the feature importances from Random Forest Classifier
importances_pca_kbest_rf = gs_pca_kbest_rf.best_estimator_.named_steps['rf_clf'].feature_importances_
indices = numpy.argsort(importances_pca_kbest_rf)[::-1]

# Use features_selected, the features selected by SelectKBest, and not features_list
print('Feature Ranking: ')
for i in range(min(len(features_selected_pca_kbest_rf), len(importances_pca_kbest_rf))):
    print("\tfeature no. {}: {} ({})".format(i+1,features_selected_pca_kbest_rf[indices[i]],importances_pca_kbest_rf[indices[i]]))

### Test best_estimator_ with tester.test_classifier
test_classifier(best_estimator_pca_kbest_rf, my_dataset, all_features)





print('    #####################################################################    ')
gs_funion_lnr_sv = GridSearchCV(funion_lnr_sv, param_grid=params_funion_lnr_sv, cv=cv, scoring='f1')

print("Performing grid search gs_funion_lnr_sv")
print("pipeline:", [name for name, _ in funion_lnr_sv.steps])
pprint(funion_lnr_sv)
print("parameters:")
pprint(params_funion_lnr_sv)
t0 = time()
gs_funion_lnr_sv.fit(features, labels)
print("done in %0.3fs" % (time() - t0))

print("Best score: %0.3f" % gs_funion_lnr_sv.best_score_)
best_estimator_funion_lnr_sv = gs_funion_lnr_sv.best_estimator_
print("Best estimator pipeline: ", gs_funion_lnr_sv.best_estimator_)
print("Best parameters set: ")
best_parameters_funion_lnr_sv = gs_funion_lnr_sv.best_estimator_.get_params()
for param_name in sorted(params_funion_lnr_sv.keys()):
    print("\t%s: %r" % (param_name, best_parameters_funion_lnr_sv[param_name]))

pred_funion_lnr_sv = gs_funion_lnr_sv.predict(features_test)
print('Precision:', precision_score(labels_test, pred_funion_lnr_sv))
print('Recall:', recall_score(labels_test, pred_funion_lnr_sv))
print('F1 Score:', f1_score(labels_test, pred_funion_lnr_sv))

kbest = best_estimator_funion_lnr_sv.named_steps['f_union'].get_params()['kbest']
print('kbest.scores_: \n', kbest.scores_)
print('kbest.pvalues_: \n', kbest.pvalues_)
print('kbest.get_params(): \n', kbest.get_params())

print('Feature Scores and PValues: ')
for f, score, pval in zip(all_features[1:], kbest.scores_, kbest.pvalues_):
    print("\tfeature {} : ( score: {}, pval: {} )".format(f, score ,pval))

features_selected_funion_lnr_sv = [all_features[i+1] for i in kbest.get_support(indices=True)]
print('kbest selected features_selected_funion_lnr_sv: \n', features_selected_funion_lnr_sv)


### Test best_estimator_ with tester.test_classifier
test_classifier(best_estimator_funion_lnr_sv, my_dataset, all_features)




# print('    #####################################################################    ')
# gs_pca_kbest_lnr_sv = GridSearchCV(pipe_pca_kbest_lnr_sv, param_grid=params_pca_kbest_lnr_sv, cv=cv, scoring='f1')
#
# print("Performing grid search pca_kbest_lnr_sv")
# print("pipeline:", [name for name, _ in pipe_pca_kbest_lnr_sv.steps])
# print("parameters:")
# pprint(params_pca_kbest_lnr_sv)
# t0 = time()
# gs_pca_kbest_lnr_sv.fit(features, labels)
# print("done in %0.3fs" % (time() - t0))
#
# print("Best score: %0.3f" % gs_pca_kbest_lnr_sv.best_score_)
# best_estimator_pca_kbest_lnr_sv = gs_pca_kbest_lnr_sv.best_estimator_
# print("Best estimator pipeline: ", gs_pca_kbest_lnr_sv.best_estimator_)
# print("Best parameters set: ")
# best_parameters_pca_kbest_lnr_sv = gs_pca_kbest_lnr_sv.best_estimator_.get_params()
# for param_name in sorted(params_pca_kbest_lnr_sv.keys()):
#     print("\t%s: %r" % (param_name, best_parameters_pca_kbest_lnr_sv[param_name]))
#
# pred_pca_kbest_lnr_sv = gs_pca_kbest_lnr_sv.predict(features_test)
# print('Precision:', precision_score(labels_test, pred_pca_kbest_lnr_sv))
# print('Recall:', recall_score(labels_test, pred_pca_kbest_lnr_sv))
# print('F1 Score:', f1_score(labels_test, pred_pca_kbest_lnr_sv))
#
# kbest = gs_pca_kbest_lnr_sv.best_estimator_.named_steps['f_kbest']
# print('kbest.scores_: \n', kbest.scores_)
# print('kbest.pvalues_: \n', kbest.pvalues_)
# print('kbest.get_params(): \n', kbest.get_params())
#
# features_selected_pca_kbest_lnr_sv = [all_features[i+1] for i in kbest.get_support(indices=True)]
# print('features_selected_pca_kbest_lnr_sv: \n', features_selected_pca_kbest_lnr_sv)
#
# ### Test best_estimator_ with tester.test_classifier
# test_classifier(best_estimator_pca_kbest_lnr_sv, my_dataset, all_features)




print('    #####################################################################    ')


#########################################################################
#
#  Best Results Options:
#  Both Option 1 and 2 Gives Same F1, Precision and Recall Scores.
#  Option 1 had GridSearchCV.best_score_ (0.314) than 2 (0.260)
#
#########################################################################
#########################################################################
# Option 1 :: pca_kbest_rf
#########################################################################
# est_pipe_pca_kbest_rf = [
#       ('f_scaler_1', f_minmaxscaler),
#       ('dim_reduc', dim_reduc),
#       ('f_scaler_2', f_minmaxscaler),
#       ('f_kbest', f_kbest),
#       ('rf_clf', rf_clf)]
# pipe_pca_kbest_rf = Pipeline(est_pipe_pca_kbest_rf)
#
# Best score: 0.314
# Best parameters set:
#       dim_reduc__n_components: 20
#       f_kbest__k: 7
#       f_kbest__score_func: <function chi2 at 0x108b80b70>
#       rf_clf__criterion: 'gini'
#       rf_clf__max_depth: 20
#       rf_clf__n_estimators: 5
#
# Precision: 1.0
# Recall: 0.8
# F1 Score: 0.888888888889
#
# Feature Ranking:
# 	     feature no. 1: other (0.25657832345323267)
# 	     feature no. 2: deferral_payments (0.24191669636447494)
# 	     feature no. 3: total_payments (0.128595092401951)
# 	     feature no. 4: exercised_stock_options (0.11711177258717469)
# 	     feature no. 5: bonus (0.11084087045516852)
# 	     feature no. 6: loan_advances (0.08755208887176714)
# 	     feature no. 7: salary (0.05740515586623096)
#########################################################################

#########################################################################
# Option 2 :: kbest_pca_rf
#########################################################################
# est_pipe_kbest_pca_rf = [
#       ('f_scaler_1', f_minmaxscaler),
#       ('f_kbest', f_kbest),
#       ('dim_reduc', dim_reduc),
#       ('rf_clf', rf_clf)]
# pipe_kbest_pca_rf = Pipeline(est_pipe_kbest_pca_rf)
#
# Best score: 0.260
# Best parameters set:
#       dim_reduc__n_components: 5
#       f_kbest__k: 20
#       f_kbest__score_func: <function chi2 at 0x108b80b70>
#       rf_clf__criterion: 'entropy'
#       rf_clf__max_depth: 20
#       rf_clf__n_estimators: 5
#
# Precision: 1.0
# Recall: 0.8
# F1 Score: 0.888888888889
#
# Feature Ranking:
#       feature no. 1: salary (0.3723075690037799)
#       feature no. 2: deferral_payments (0.23866102552083235)
#       feature no. 3: bonus (0.17434441095328976)
#       feature no. 4: total_payments (0.10805908541163647)
#       feature no. 5: loan_advances (0.10662790911046145)
#######################################################################


#########################################################################
#
#  Export Option 1 Best Estimator and Features via
#  tester.dump_classifier_and_data
#
#########################################################################
my_clf = best_estimator_funion_lnr_sv
my_feature_list = all_features
#
#
# from tester import dump_classifier_and_data, test_classifier
# ### Dump pickle files
dump_classifier_and_data(my_clf, my_dataset, my_feature_list)
# ### Perform Test on Classifier
test_classifier(my_clf, my_dataset, my_feature_list)

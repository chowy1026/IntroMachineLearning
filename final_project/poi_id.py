#!/usr/bin/python3

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


print("Number of Data Points: %d " % (tot_data_points))
print("Number of Features: %d " % (tot_features))


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


print("Number of POIs: %d " % (int_poi_count))
print("Number of non-POIs: %d " % (int_non_poi_count))


quantifiable_poi = get_quantifiable_count('poi')
quantifiable_email = get_quantifiable_count('email_address')

### No missing data for POI feature - good
print("Non-NaN 'poi': %d " % (quantifiable_poi))
print("Percentage of Non-NaN 'poi': %0.3f " % (float(quantifiable_poi)/float(tot_data_points)))

### Some missing data for email_address feature
print("Non-NaN 'email_address': %d " % (quantifiable_email))
print("Percentage of Non-NaN 'email_address': %0.3f " % (float(quantifiable_email)/float(tot_data_points)))


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

print("%s got the highest count of missing data with %d features missing." % (max(NaN_feature_count, key=NaN_feature_count.get), max(NaN_feature_count.values())))
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

### Uncomment below to plot charts
# exploration_plot(enron_data, 'total_stock_value', 'total_payments')

### Task 2: Remove outliers
enron_data.pop("TOTAL", 0) # Remove outlier predetermined from mini-project
enron_data.pop('LOCKHART EUGENE E', 0) # has no data
enron_data.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Obviously travel agency has nothing to do with Enron

### Uncomment below to plot charts
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
print('%d all_features: %r' % (len(all_features), all_features))
### Store to my_dataset for easy export below.
my_dataset = enron_data


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
names = []
def add_estimator_pipe(name, estimator, pipe):
    names.append(name)
    estimators.append(estimators)
    pipes.append(pipe)


###0 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, linear svc
est_pipe_pca_kbest_lnr_sv = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('lnr_sv_clf', lnr_sv_clf)]
pipe_pca_kbest_lnr_sv = Pipeline(est_pipe_pca_kbest_lnr_sv)
add_estimator_pipe("PCA_KBest_LinearSVC", est_pipe_pca_kbest_lnr_sv, pipe_pca_kbest_lnr_sv)

###1 Scaler, kbest, pca, linear svc
est_pipe_kbest_pca_lnr_sv = [('f_scaler_1', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('lnr_sv_clf', lnr_sv_clf)]
pipe_kbest_pca_lnr_sv = Pipeline(est_pipe_kbest_pca_lnr_sv)
add_estimator_pipe("KBest_PCA_LinearSVC", est_pipe_kbest_pca_lnr_sv, pipe_kbest_pca_lnr_sv)

###2 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, rbf svc
est_pipe_pca_kbest_rbf_sv = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('rbf_sv_clf', rbf_sv_clf)]
pipe_pca_kbest_rbf_sv = Pipeline(est_pipe_pca_kbest_rbf_sv)
add_estimator_pipe("PCA_KBest_RBFSCV", est_pipe_pca_kbest_rbf_sv, pipe_pca_kbest_rbf_sv)

###3 Scaler, kbest, pca, rbf svc
est_pipe_kbest_pca_rbf_sv = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('rbf_sv_clf', rbf_sv_clf)]
pipe_kbest_pca_rbf_sv = Pipeline(est_pipe_kbest_pca_rbf_sv)
add_estimator_pipe("KBest_PCA_RBFSCV", est_pipe_kbest_pca_rbf_sv, pipe_kbest_pca_rbf_sv)

###4 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, decision tree classifier
est_pipe_pca_kbest_dt = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('dt_clf', dt_clf)]
pipe_pca_kbest_dt = Pipeline(est_pipe_pca_kbest_dt)
add_estimator_pipe("PCA_KBest_DecisionTree", est_pipe_pca_kbest_dt, pipe_pca_kbest_dt)

###5 Scaler, kbest, pca, decision tree classifier
est_pipe_kbest_pca_dt = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('dt_clf', dt_clf)]
pipe_kbest_pca_dt = Pipeline(est_pipe_kbest_pca_dt)
add_estimator_pipe("KBest_PCA_DecisionTree", est_pipe_kbest_pca_dt, pipe_kbest_pca_dt)

###6 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, gaussian naive bayes classifier
est_pipe_pca_kbest_nb = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('nb_clf', nb_clf)]
pipe_pca_kbest_nb = Pipeline(est_pipe_pca_kbest_nb)
add_estimator_pipe("PCA_KBest_DecisionTree", est_pipe_pca_kbest_nb, pipe_pca_kbest_nb)

###7 Scaler, kbest, pca, gaussian naive bayes classifier
est_pipe_kbest_pca_nb = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('nb_clf', nb_clf)]
pipe_kbest_pca_nb = Pipeline(est_pipe_kbest_pca_nb)
add_estimator_pipe("KBest_PCA_DecisionTree", est_pipe_kbest_pca_nb, pipe_kbest_pca_nb)

###8 Scaler_1, pca, scaler_2[to remove negative values from PCA], kbest, random forest classifier
est_pipe_pca_kbest_rf = [('f_scaler_1', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('f_scaler_2', f_stdscaler), ('rf_clf', rf_clf)]
pipe_pca_kbest_rf = Pipeline(est_pipe_pca_kbest_rf)
add_estimator_pipe("PCA_KBest_RandomForest", est_pipe_pca_kbest_rf, pipe_pca_kbest_rf)

###9 Scaler, kbest, pca, random forest classifier
est_pipe_kbest_pca_rf = [('f_scaler_1', f_minmaxscaler),  ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('f_scaler_2', f_stdscaler), ('rf_clf', rf_clf)]
pipe_kbest_pca_rf = Pipeline(est_pipe_kbest_pca_rf)
add_estimator_pipe("KBest_PCA_RandomForest", est_pipe_kbest_pca_rf, pipe_kbest_pca_rf)

###10 Scaler, featureUnion[pca & kbest], linear svc
est_funion_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('lnr_sv_clf', lnr_sv_clf)]
funion_lnr_sv = Pipeline(est_funion_lnr_sv)
add_estimator_pipe("[KBest_Union_PCA]_LinearSVC", est_funion_lnr_sv, funion_lnr_sv)

###11 Scaler, featureUnion[pca & kbest], rbf svc
est_funion_rbf_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rbf_sv_clf', rbf_sv_clf)]
funion_rbf_sv = Pipeline(est_funion_rbf_sv)
add_estimator_pipe("[KBest_Union_PCA]_RBFSCV", est_funion_rbf_sv, funion_rbf_sv)

###12 Scaler, featureUnion[pca & kbest], random forest classifier
est_funion_rf = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rf_clf', rf_clf)]
funion_rf = Pipeline(est_funion_rf)
add_estimator_pipe("[KBest_Union_PCA]_RandomForest", est_funion_rf, funion_rf)

# print(len(estimators), " estimators :: ", estimators)
# print(len(pipes), " pipes :: ", pipes)


### Use featureFormat.train_test_split to get training and testing sets to perform
### quick fit with the above estimators, and get classification_reports
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Quick run with training set to roughly select the model with highest
### f1, precision and recall scores

c = 0
for nm, estimator, pipe in zip(names, estimators, pipes):
    print("\n=================================================================")
    # for name, est in estimator:
    #     print("\nParameters for ", name, " :: \n", est.get_params().keys())


    pipe.fit(features_train, labels_train)
    labels_pred = pipe.predict(features_test)
    clf_reports = classification_report(labels_test, labels_pred)
    print("Classification Report for %s " %(nm))
    print("\n %s" % (clf_reports))
    c += 1
    print("=================================================================\n")







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

# With some trials and errors, I noticed some estimator works better than order,
# and some orders work bettter than others.
# Generally below gives better performances:
# - PCA before SelectKBest
# - Random Forest is better than DecisionTree
# - LinearSVC is better than RBFSCV
# - performances is better when MinMaxScaler is mixed with StandardScaler
# (However StandardScaler wont work with GridSearchCV below)



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

###12 Scaler, featureUnion[pca & kbest], random forest classifier
est_funion_rf = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rf_clf', rf_clf)]
funion_rf = Pipeline(est_funion_rf)


### Setting up Param_Grids for GridSearchCV ####
params_funion_lnr_sv = dict(
f_union__pca_scaler__pca__n_components = [None, 22],
f_union__kbest__k = [20, 17],
f_union__kbest__score_func = [f_classif, chi2],
lnr_sv_clf__C = [1, 10, 100]
)
params_funion_rf = dict(
f_union__pca_scaler__pca__n_components = [None, 22],
f_union__kbest__k = [20, 17],
f_union__kbest__score_func = [f_classif, chi2],
rf_clf__n_estimators = [2, 5, 10],
rf_clf__criterion = ['gini', 'entropy'],
rf_clf__max_depth = [15, 20, 25]
)
params_pipe_pca_kbest_rf = dict(
dim_reduc__n_components = [17, 15, 12],
f_kbest__k = [7, 5, 3, 2],
f_kbest__score_func = [f_classif, chi2],
rf_clf__n_estimators = [4, 5, 6],
rf_clf__criterion  = ['gini', 'entropy'],
rf_clf__max_depth = [20, 22, 25],
)
#######
params_pipe_kbest_pca_rf = dict(
f_kbest__k = [20, 15, 10],
f_kbest__score_func = [f_classif, chi2],
dim_reduc__n_components = [7, 5, 3],
rf_clf__n_estimators = [5, 12],
rf_clf__criterion = ['gini', 'entropy'],
rf_clf__max_depth = [20, 30]
)
params_pipe_pca_kbest_lnr_sv = dict(
dim_reduc__n_components = [20, 15, 10],
f_kbest__k = [7, 5, 3],
f_kbest__score_func = [f_classif, chi2],
lnr_sv_clf__C = [0.001, 0.01, 0.1, 1]
)

grid_search_estimators = [est_funion_lnr_sv, est_funion_rf, est_pipe_pca_kbest_rf, est_pipe_kbest_pca_rf]
grid_search_pipes = [funion_lnr_sv, funion_rf, pipe_pca_kbest_rf, pipe_kbest_pca_rf]
grid_search_params = [params_funion_lnr_sv, params_funion_rf, params_pipe_pca_kbest_rf, params_pipe_kbest_pca_rf]
grid_search_names = ['[KBest_Union_PCA_Scaler]_LinearSVC', '[KBest_Union_PCA_Scaler]_RandomForest', 'PCA_Scaler_KBest_RandomForest', 'KBest_PCA_RandomForest']
grid_search_dict_results = [] # Empty Results List to store grid_search Results

cv = StratifiedShuffleSplit(n_splits=100, random_state=42)

### This is the same function from tester.test_classifier to return scores in
### two different formats - dictionary and list.
### I made this because I assume minimum changes shall be made on tester.py as
### it wont be evaluated.
def computeScores(clf):
    PERF_FORMAT_STRING = "\
    \tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
    RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
    \tFalse negatives: {:4d}\tTrue negatives: {:4d}"

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    tester_cv = StratifiedShuffleSplit(1000, random_state = 42)
    # for train_idx, test_idx in cv: ### This only works w. older ver of sklearn
    ### Replace with the line below for StratifiedShuffleSplit of sklearn v0.18
    for train_idx, test_idx in tester_cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break

    scores_dict = {}
    scores_list = []

    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        scores_list.append(accuracy)
        scores_list.append(precision)
        scores_list.append(recall)
        scores_list.append(f1)
        scores_list.append(f2)
        scores_dict['accuracy'] = accuracy
        scores_dict['precision'] = precision
        scores_dict['recall'] = recall
        scores_dict['f1'] = f1
        scores_dict['f2'] = f2

        ### Print string like from tester
        print(clf)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")

    return scores_dict, scores_list



def get_pipeline_parts(best_estimator):
    f_union = None
    kbest = None
    try:
        f_union = best_estimator.named_steps['f_union']
        if f_union is not None:
            try:
                kbest = best_estimator.named_steps['f_union'].get_params()['kbest']
            except:
                kbest = None
                print('\nNo SelectKBest Found in FeatureUnion.')
    except:
        f_union = None
        print('\nNo FeatureUnion Found.')
        try:
            kbest = best_estimator.named_steps['f_kbest']
        except:
            kbest = None
            print('\nNo SelectKBest Found in Pipeline.')

    isSVC = False
    clf = None
    ### See if classifier an SVC or RandomForest/DecisionTree
    try:
        clf = best_estimator.named_steps['lnr_sv_clf']
        isSVC = True if clf is not None else False
    except:
        try:
            clf = best_estimator.named_steps['rf_clf']
            isSVC = True if clf is None else False
        except:
            print('\nNo Classifier Found.')

    return f_union, kbest, isSVC, clf


def kbest_props(kbest):
    print(' ')
    print('kbest.scores_: \n%s' % kbest.scores_)
    print('kbest.pvalues_: \n%s' % kbest.pvalues_)
    print('kbest.get_params(): \n%s' % kbest.get_params())

    print(' ')
    print('Feature Scores and PValues: ')
    for f, score, pval in zip(all_features[1:], kbest.scores_, kbest.pvalues_):
        print("\t\tfeature %s : ( score: %0.5f, pval: %0.5f ) " % (f, score ,pval))

    print(' ')
    kbest_features_selected = [all_features[i+1] for i in kbest.get_support(indices=True)]
    print('%d kbest features_selected: \n%s' % (len(kbest_features_selected),  kbest_features_selected))
    return kbest_features_selected

def selected_features_importance(clf, features):
    ### ONLY for clf that are RandomForest or DecisionTree Classifiers
    importances = clf.feature_importances_
    indices = numpy.argsort(importances)[::-1]
    print(' ')
    print(len(importances), " importances:\n ", importances)
    print(len(indices), " indices:\n ", indices)
    print(' ')
    print('Feature Ranking by Importance: ')

    for i in range(min(len(features), len(importances))):
        print("\t\tfeature no. %d: %s (%0.5f)" % (i+1 , features[indices[i]], importances[indices[i]]))


def best_gridsearchcv_fit(pipe, param_grid):
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='f1')

    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    pprint(param_grid)
    t0 = time()
    grid_search.fit(features, labels)
    print("done in %0.3fs" % (time() - t0))

    print(' ')
    print("Best score: %0.3f" % grid_search.best_score_)

    best_estimator = grid_search.best_estimator_
    print(' ')
    print("Best estimator pipeline: ", best_estimator)
    print(' ')
    print("Best parameters set: ")
    best_parameters = best_estimator.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t\t%s: %r" % (param_name, best_parameters[param_name]))

    labels_pred = grid_search.predict(features_test)
    print(' ')
    print('Precision: %0.3f ' % precision_score(labels_test, labels_pred))
    print('Recall: %0.3f ' % recall_score(labels_test, labels_pred))
    print('F1 Score: %0.3f ' % f1_score(labels_test, labels_pred))

    return best_estimator



### This is a function to perform GridSearchCV fitting and return
### the GridSearchCV.best_estimator
def do_gridsearchcv(pipe, param_grid):

    best_estimator = best_gridsearchcv_fit(pipe, param_grid)

    f_union, kbest, isSVC, clf = get_pipeline_parts(best_estimator)

    features = None
    if kbest is not None:
        features = kbest_props(kbest)
        if f_union is not None:
            features = features + all_features[1:]

    ### Get feature importance if clf is RandomForest or DecisionTree.
    if isSVC == False:  ###
        selected_features_importance(clf, features)

    gs_dict, gs_list  = computeScores(best_estimator)
    gs_dict['name'] = name
    gs_dict['best_estimator'] = best_estimator
    grid_search_dict_results.append(gs_dict)



i = 0 # As counter
n = 9 # Limit, if don't want to fit the whole list


### Loop through the list of grid_search_pipes and perform GridSearchCV
### Append results (dictionary of name, pipeline, performance scores) to
### grid_search_results list
for name, pipe, param_grid in zip(grid_search_names, grid_search_pipes, grid_search_params):
    print('    #####################################################################    ')
    print("      Performing Grid Search on %s" % name)
    print('    #####################################################################    ')
    print(' ')

    do_gridsearchcv(pipe, param_grid)

    print('    #####################################################################\n\n\n    ')


print('\n ')
print('    #################   Model with Highest Accuracy   ###################  ')
highest_acc = sorted(grid_search_dict_results, key=lambda k: k['accuracy'], reverse=True)
pprint(highest_acc[0])
pprint(highest_acc[0]['best_estimator'])

print('\n ')
print('    #################   Model with Highest F1 Score   ###################  ')
highest_f1 = sorted(grid_search_dict_results, key=lambda k: k['f1'], reverse=True)
pprint(highest_f1[0])
pprint(highest_f1[0]['best_estimator'])

print('\n ')
print('    #################   Model with Highest F2 Score   ###################  ')
highest_f2 = sorted(grid_search_dict_results, key=lambda k: k['f2'], reverse=True)
# highest_f2 = sorted(grid_search_dict_results, key=itemgetter('f2'), reverse=True)
pprint(highest_f2[0])
pprint(highest_f2[0]['best_estimator'])

print('\n ')
print('    ##############   Model with Highest Precision Score   ###############  ')
highest_precision = sorted(grid_search_dict_results, key=lambda k: k['precision'], reverse=True)
# highest_precision = sorted(grid_search_dict_results, key=itemgetter('precision'), reverse=True)
pprint(highest_precision[0])
pprint(highest_precision[0]['best_estimator'])

print('\n ')
print('    #################   Model with Highest Recall Score ################# ')
highest_recall = sorted(grid_search_dict_results, key=lambda k: k['recall'], reverse=True)
# highest_recall = sorted(grid_search_dict_results, key=itemgetter('recall'), reverse=True)
pprint(highest_recall[0])
pprint(highest_recall[0]['best_estimator'])

print('\n ')
print('    ####  Model with Highest F1, Precision, Recall, Accuracy Score  ##### ')
sorted_grid_search_dict_results = sorted(grid_search_dict_results, key=lambda k: (k['f1'], k['precision'], k['recall'], k['accuracy']), reverse=True)
pprint(sorted_grid_search_dict_results[0])
pprint(sorted_grid_search_dict_results[0]['best_estimator'])

### Submit / Export files for tester.py
print('\n ')
print('\n ')
my_clf = sorted_grid_search_dict_results[0]['best_estimator']
my_feature_list = all_features
from tester import dump_classifier_and_data, test_classifier
### Dump pkl files
dump_classifier_and_data(my_clf, my_dataset, my_feature_list)
### Run my_clf, my_dataset and my_feature_list against tester.test_classifier
print('   ###########   Final Results from Best Estimator Options    ###########   ')
test_classifier(my_clf, my_dataset, my_feature_list)

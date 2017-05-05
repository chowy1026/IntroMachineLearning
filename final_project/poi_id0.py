

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


print("Number of Data Points: ", str(tot_data_points))
print("Number of Features: ", str(tot_features))


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


print("Number of POIs: ", str(int_poi_count))
print("Number of non-POIs: ", str(int_non_poi_count))


quantifiable_poi = get_quantifiable_count('poi')
quantifiable_email = get_quantifiable_count('email_address')

### No missing data for POI feature - good
print("Non-NaN 'poi'", str(quantifiable_poi))
print("Percentage of Non-NaN 'poi'", str(float(quantifiable_poi)/float(tot_data_points)))

### Some missing data for email_address feature
print("Non-NaN 'email_address'", str(quantifiable_email))
print("Percentage of Non-NaN 'email_address'", str(float(quantifiable_email)/float(tot_data_points)))


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
print(orig_feature_count_density)


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

print("%s got the highest count of missing data at %s" % (max(NaN_feature_count, key=NaN_feature_count.get), str(max(NaN_feature_count.values()))))
print(sorted_NaN_feature_count)


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
        print(type(point))
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


### Task 3: Create new feature(s)
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
print(new_feature_count_density)

### New All Original and New Feature List
all_features = bool_features + numeric_features + new_features

### Store to my_dataset for easy export below.
my_dataset = enron_data


### Extract features and labels from dataset for local testing
print('all_features :: ', all_features)
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

### Build Estimators for Pipeline
f_minmaxscaler = preprocessing.MinMaxScaler()
f_kbest = feature_selection.SelectKBest()
dim_reduc = PCA(svd_solver='randomized', random_state=42)
lnr_sv_clf = SVC(kernel="linear", random_state=42)
rbf_sv_clf = SVC(kernel="rbf", random_state=42)
dt_clf = DecisionTreeClassifier(random_state=42)
nb_clf = GaussianNB()
rf_clf = RandomForestClassifier(random_state=42)
f_union = FeatureUnion([("kbest", f_kbest), ("pca", dim_reduc)])


### First I tried to perform quick fit and score for different order and combinations of estimators and classifiers with pipeline and feature_union.

### Scaler, [pca then kbest / kbest then pca], linear svc
est_pipe_lnr_sv = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('lnr_sv_clf', lnr_sv_clf)]
est_pipe_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('lnr_sv_clf', lnr_sv_clf)]
pipe_lnr_sv = Pipeline(est_pipe_lnr_sv)

### Scaler, [pca then kbest / kbest then pca], rbf svc
est_pipe_rbf_sv = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('rbf_sv_clf', rbf_sv_clf)]
# est_pipe_rbf_sv = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('rbf_sv_clf', rbf_sv_clf)]
pipe_rbf_sv = Pipeline(est_pipe_rbf_sv)

### Scaler, [pca then kbest / kbest then pca], decision tree classifier
est_pipe_dt = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('dt_clf', dt_clf)]
# est_pipe_dt = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('dt_clf', dt_clf)]
pipe_dt = Pipeline(est_pipe_dt)

### Scaler, [pca then kbest / kbest then pca], gaussian naive bayes classifier
est_pipe_nb = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('nb_clf', nb_clf)]
# est_pipe_nb = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('nb_clf', nb_clf)]
pipe_nb = Pipeline(est_pipe_nb)

### Scaler, [pca then kbest / kbest then pca], random forest classifier
est_pipe_rf = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('rf_clf', rf_clf)]
# est_pipe_rf = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('rf_clf', rf_clf)]
pipe_rf = Pipeline(est_pipe_rf)

### Scaler, featureUnion[pca & kbest], linear svc
est_funion_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('lnr_sv_clf', lnr_sv_clf)]
funion_lnr_sv = Pipeline(est_funion_lnr_sv)

### Scaler, featureUnion[pca & kbest], rbf svc
est_funion_rbf_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rbf_sv_clf', rbf_sv_clf)]
funion_rbf_sv = Pipeline(est_funion_rbf_sv)

### Scaler, featureUnion[pca & kbest], random forest classifier
est_funion_rf = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('rf_clf', rf_clf)]
funion_rf = Pipeline(est_funion_rf)

estimators = [est_pipe_lnr_sv, est_pipe_rbf_sv, est_pipe_dt, est_pipe_nb, est_pipe_rf, est_funion_lnr_sv, est_funion_rbf_sv, est_funion_rf]
pipes = [pipe_lnr_sv, pipe_rbf_sv, pipe_dt, pipe_nb, pipe_rf, funion_lnr_sv, funion_rbf_sv, funion_rf]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Quick run with training set to roughly select the model with highest f1, precision and recall scores
for estimator, pipe in zip(estimators, pipes):
    print("\n=================================================================")
    for name, est in estimator:
        print("\nParameters for ", name, " :: \n", est.get_params().keys())


    pipe.fit(features_train, labels_train)
    labels_pred = pipe.predict(features_test)
    clf_reports = classification_report(labels_test, labels_pred)
    print("\nClassification Report for ", name, "\n", clf_reports)
    print("\n=================================================================")


################################################################################
# The 3 following gives the best scores (See attached text files):
# 1) scaler, pca, kbest, random forest classifier (f1: 0.88, recall: 0.91, precision: 0.92)
# 2) scales, kbest, pca, random forest classifier (f1: 0.88, recall: 0.88, precision: 0.87)
# 3) scales, f_union(pca, kbest), linear svc (f1: 0.88, recall: 0.88, precision: 0.87)
################################################################################

### Scaler, pca, kbest, random forest classifier
est_pipe_pca_kbest_rf = [('f_scaler', f_minmaxscaler), ('dim_reduc', dim_reduc), ('f_kbest', f_kbest), ('rf_clf', rf_clf)]
pipe_pca_kbest_rf = Pipeline(est_pipe_pca_kbest_rf)

### Scaler, kbest, pca, random forest classifier
est_pipe_kbest_pca_rf = [('f_scaler', f_minmaxscaler), ('f_kbest', f_kbest), ('dim_reduc', dim_reduc), ('rf_clf', rf_clf)]
pipe_kbest_pca_rf = Pipeline(est_pipe_kbest_pca_rf)

### Scaler, featureUnion[pca & kbest], linear svc
est_funion_lnr_sv = [('f_scaler', f_minmaxscaler), ('f_union', f_union), ('lnr_sv_clf', lnr_sv_clf)]
funion_lnr_sv = Pipeline(est_funion_lnr_sv)


#### Setting Param_Grids for GridSearchCV ####
#########
# params_pipe_lnr_sv = dict(
# dim_reduc__n_components = [3, 5, 7, 10, 13, 17],
# f_kbest__k = [1, 3, 5, 10, 13],
# f_kbest__score_func = [f_classif, chi2],
# lnr_sv_clf__C = [0.01, 1, 10, 100]
# )
# params_pipe_rbf_sv = dict(
# dim_reduc__n_components = [3, 5, 7, 10, 13, 17],
# f_kbest__k = [1, 3, 5, 10, 13],
# f_kbest__score_func = [f_classif, chi2],
# rbf_sv_clf__C = [0.01, 1, 10, 100],
# rbf_sv_clf__gamma = [0.001, 0.01, 0.1]
# )
# params_pipe_dt = dict(
# dim_reduc__n_components = [3, 5, 7, 10, 13, 17],
# f_kbest__k = [1, 3, 5, 10, 13],
# f_kbest__score_func = [f_classif, chi2],
# dt_clf__criterion = ['gini', 'entropy'],
# dt_clf__max_depth = [10, 15, 20, 25, 30]
# )
# params_pipe_nb = dict(
# dim_reduc__n_components = [3, 5, 7, 10, 13, 17],
# f_kbest__k = [1, 3, 5, 10, 13],
# f_kbest__score_func = [f_classif, chi2]
# )
params_pipe_rf = dict(
dim_reduc__n_components = [3, 5, 7, 10, 13, 17],
f_kbest__k = ['all', 1, 3, 5, 10, 13],
f_kbest__score_func = [f_classif, chi2],
rf_clf__n_estimators = [2, 5, 7, 10],
rf_clf__criterion = ['gini', 'entropy'],
rf_clf__max_depth = [10, 15, 20, 25, 30]
)
params_funion_lnr_sv = dict(
f_union__pca__n_components = [3, 5, 7, 10, 13, 17],
f_union__kbest__k = [1, 3, 5, 10, 13],
f_union__kbest__score_func = [f_classif, chi2],
lnr_sv_clf__C = [0.001]
)
# params_funion_rbf_sv = dict(
# f_union__pca__n_components = [3, 5, 7, 10, 13, 17],
# f_union__kbest__k = [1, 3, 5, 10, 13],
# f_union__kbest__score_func = [f_classif, chi2],
# rbf_sv_clf__C = [0.01, 1, 10, 100],
# rbf_sv_clf__gamma = [0.001, 0.01, 0.1]
# )
# params_funion_rf = dict(
# f_union__pca__n_components = [3, 5, 7, 10, 13, 17],
# f_union__kbest__k = [1, 3, 5, 10, 13],
# f_union__kbest__score_func = [f_classif, chi2],
# rf_clf__n_estimators = [2, 5, 7, 10],
# rf_clf__criterion = ['gini', 'entropy'],
# rf_clf__max_depth = [10, 15, 20, 25, 30]
# )

# params = [params_pipe_lnr_sv, params_pipe_rbf_sv, params_pipe_dt, params_pipe_nb, params_pipe_rf, params_funion_lnr_sv, params_funion_rbf_sv, params_funion_rf]

grid_search_estimators = [est_pipe_pca_kbest_rf, est_pipe_kbest_pca_rf, est_funion_lnr_sv]
grid_search_pipes = [pipe_pca_kbest_rf, pipe_kbest_pca_rf, funion_lnr_sv]
grid_search_params = [params_pipe_rf, params_pipe_rf, params_funion_lnr_sv]

cv = StratifiedShuffleSplit(n_splits=100, random_state=42, test_size=0.1)

for estimator, pipe, paramgrid in zip(grid_search_estimators, grid_search_pipes, grid_search_params):
    print("\n=================================================================")
    grid_search = GridSearchCV(pipe, param_grid=paramgrid, cv=cv, scoring='f1', verbose=10, n_jobs=-1, error_score=0)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    pprint(paramgrid)
    t0 = time()
    grid_search.fit(features, labels)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    best_estimator = grid_search.best_estimator_
    print("Best estimator pipeline: ", best_estimator)
    print("Best parameters set:")
    best_parameters = best_estimator.get_params()
    for param_name in sorted(paramgrid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    pred = grid_search.predict(features_test)
    print('Precision:', precision_score(labels_test, pred))
    print('Recall:', recall_score(labels_test, pred))
    print('F1 Score:', f1_score(labels_test, pred))
    print("\n=================================================================")
    # grid_search_best = grid_search.best_estimator_
    # features_selected=[all_features[i+1] for i in grid_search_best.named_steps['f_union__kbest'].get_support(indices=True)]


exit()

grid_search = GridSearchCV(pipe_pca_kbest_rf, param_grid=params_pipe_rf, cv=cv, scoring='f1', verbose=10, error_score=0)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe_pca_kbest_rf.steps])
print("parameters:")
pprint(params_pipe_rf)
t0 = time()
grid_search.fit(features, labels)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
best_estimator = grid_search.best_estimator_
print("Best estimator pipeline: ", best_estimator)
print("Best parameters set: ")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(params_pipe_rf.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

pred = grid_search.predict(features_test)
print('Precision:', precision_score(labels_test, pred))
print('Recall:', recall_score(labels_test, pred))
print('F1 Score:', f1_score(labels_test, pred))



# Access the SelectKBest features selected

# create a new list that contains the features selected by SelectKBest
# in the optimal model selected by GridSearchCV
features_selected=[all_features[i+1] for i in best_estimator.named_steps['f_kbest'].get_support(indices=True)]

# Access the feature importances

# The step in the pipeline for the Decision Tree Classifier is called 'DTC'
# that step contains the feature importances
importances = best_estimator.named_steps['rf_clf'].feature_importances_
indices = numpy.argsort(importances)[::-1]

# Use features_selected, the features selected by SelectKBest, and not features_list
print('Feature Ranking: ')
for i in range(len(features_selected)):
    print("feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]]))

# print("best_estimator_", "::", grid_search.best_estimator_)
# print("best_score_", "::", grid_search.best_score_)
# print("best_params_", "::", grid_search.best_params_)

my_clf = best_estimator
my_feature_list = features_selected
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# estimators = [('reduce_dim', PCA()), ('clf', SVC())]
# pipe = Pipeline(estimators)
# pipe
# Pipeline(steps=[('reduce_dim', PCA(copy=True, iterated_power='auto',
# n_components=None, random_state=None, svd_solver='auto', tol=0.0,
# whiten=False)), ('clf', SVC(C=1.0, cache_size=200, class_weight=None,
# coef0=0.0, decision_function_shape=None, degree=3, gamma='auto',
# kernel='rbf', max_iter=-1, probability=False, random_state=None,
# shrinking=True, tol=0.001, verbose=False))])

# ### Task 4: Try a varity of classifiers
# ### Please name your classifier clf for easy export below.
# ### Note that if you want to do PCA or other multi-stage operations,
# ### you'll need to use Pipelines. For more info:
# ### http://scikit-learn.org/stable/modules/pipeline.html
#
# # Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
my_clf = best_estimator
my_feature_list = features_selected
from tester import dump_classifier_and_data
dump_classifier_and_data(my_clf, my_dataset, my_feature_list)

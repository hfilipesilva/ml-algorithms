import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift, Birch
from sklearn.model_selection import GridSearchCV
import pandas_profiling as pp
import itertools
import sys

class Entry():
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
class Dataset(BaseEstimator, TransformerMixin): 
    def __init__(self, isPrompt, option, entry):
        self.dataframe = []
        self.dataframeTest = []
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        
    def printMenu(self):
        print('1 - Read dataset')
        print('2 - Split dataset')
        print('3 - Clear output')
        print('4 - Return main menu')
        
    def fit(self, x):
        return self
        
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                
                if(self.option == '1'):
                    print('==================== Step Read dataset ====================')
                    self.dataframe = pd.read_csv(input('Enter dataset path:'), sep = ',')
                elif(self.option == '2'):
                    print('==================== Step Split dataset ====================')
                    dataframeTrain, dataframeTest = train_test_split(self.dataframe, test_size = (float(input('Enter test percentage:'))/100.0), random_state = 1)
                    dataframeTrain.to_csv(input('Enter training dataset path'), index = False, header = True)
                    dataframeTest.to_csv(input('Enter test dataset path'), index = False, header = True)
                elif(self.option == '3'):
                    clear_output()
                elif(self.option == '4'):
                    break
                    
                if(not self.isPrompt):
                    return self.dataframe
            except:
                print('An error occurred', sys.exc_info())
                
class Operation:
    def __init__(self, dataset): 
        self.dataset = dataset

class MetadataOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        
    def printMenu(self):
        print('1 - Show dataset columns')
        print('2 - Show type of columns')
        print('3 - Convert columns to lower case and remove white spaces')
        print('4 - Rename column')
        print('5 - Clear output')
        print('6 - Return main menu')
        
    def fit(self, x):
        return self
        
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                    
                if(self.option == '1'):
                    print('==================== Step show columns ====================')
                    print(x.columns)
                elif(self.option == '2'):
                    print('==================== Step show type columns ====================')
                    print(x.dtypes)
                elif(self.option == '3'):
                    print('==================== Step lower case and white space removal for dataset columns ====================')
                    x.columns = x.columns.str.strip().str.lower()
                elif(self.option == '4'):
                    print('==================== Step rename column ====================')
                    x.rename(columns = {input('Enter feature:') if self.isPrompt else self.entry.name: (input('Enter value') if self.isPrompt else self.entry.value)}, inplace = True)
                elif(self.option == '5'):
                    clear_output()
                elif(self.option == '6'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
            
class DataOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, firstOption, secondOption, entry, extraEntry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.firstOption = firstOption
        self.secondOption = secondOption
        self.entry = entry
        self.extraEntry = extraEntry
        
    def printMenu(self):
        print('1 - Dataset menu')
        print('2 - Feature menu')
        print('3 - Duplicates menu')
        print('4 - Missing values menu')
        print('5 - Scaler menu')
        print('6 - Clear output')
        print('7 - Return main menu')
        
    def printDatasetMenu(self):
        print('1 - Describe dataset')
        print('2 - Info dataset')
        print('3 - Head dataset')
        print('4 - Show dataset shape')
        print('5 - Clear output')
        print('6 - Return previous menu')
        
    def printFeatureMenu(self):
        print('1 - Describe feature')
        print('2 - Show feature value counts')
        print('3 - Show unique feature values')
        print('4 - Remove outliers from feature')
        print('5 - Show unique feature values')
        print('6 - Combine features')
        print('7 - Show correlation between features')
        print('8 - ANOVA feature selection (numerical)')
        print('9 - Chi-square feature selection (categorical)')
        print('10 - Mutual information feature selection (categorical)')
        print('11 - Encode feature (categorical)')
        print('12 - Encode feature one binary attribute (categorical)')
        print('13 - Encode feature (ordinal)')
        print('14 - Combine all features')
        print('15 - Search for feature instance')
        print('16 - Show feature median')
        print('17 - Show feature mode')
        print('18 - Show feature variance')
        print('19 - Show feature skewness')
        print('20 - Show feature kurtosis')
        print('21 - Clear output')
        print('22 - Return previous menu')
    
    def printDuplicatesMenu(self):
        print('1 - Any duplicates')
        print('2 - Show duplicated values')
        print('3 - Remove duplicated values')
        print('4 - Clear output')
        print('5 - Return previous menu')
        
    def printMissingValuesMenu(self):
        print('1 - Show total null values')
        print('2 - Remove null values')
        print('3 - Remove instance')
        print('4 - Remove feature')
        print('5 - Value imputation with mean')
        print('6 - Value imputation with median')
        print('7 - Value imputation with most frequent')
        print('8 - Value imputation with constant')
        print('9 - Clear output')
        print('10 - Return previous menu')
        
    def printScalerMenu(self):
        print('1 - Min Max Scaler from feature')
        print('2 - Standard Scaler from feature')
        print('3 - Clear output')
        print('4 - Return previous menu')
        
    def fit(self, x):
        return self
        
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.firstOption = input('Choose option:')
                
                if(self.firstOption == '1'):
                    self.executeDatasetFlow(x)
                elif(self.firstOption == '2'):
                    self.executeFeatureFlow(x)
                elif(self.firstOption == '3'):
                    self.executeDuplicatesFlow(x)
                elif(self.firstOption == '4'):
                    self.executeMissingValuesFlow(x)
                elif(self.firstOption == '5'):
                    self.executeScalerFlow(x)
                elif(self.firstOption == '6'):
                    clear_output()
                elif(self.firstOption == '7'):
                    break
                
                if(not self.isPrompt):
                    return self.dataset.dataframe
            except:
                print('An error occurred', sys.exc_info())
    
    def executeDatasetFlow(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printDatasetMenu()
                    self.secondOption = input('Choose option:')
                
                if(self.secondOption == '1'):
                    print('==================== Step describe dataset ====================')
                    print(x.describe())
                elif(self.secondOption == '2'):
                    print('==================== Step info dataset ====================')
                    print(x.info())
                elif(self.secondOption == '3'):
                    print('==================== Step head dataset ====================')
                    print(x.head(int(input('Enter top value:') if self.isPrompt else self.entry.value)))
                elif(self.secondOption == '4'):
                    print('==================== Step dataset shape ====================')
                    print(x.shape)
                elif(self.secondOption == '5'):
                    clear_output()
                elif(self.secondOption == '6'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
                
    def executeFeatureFlow(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printFeatureMenu()
                    self.secondOption = input('Choose option:')    

                if(self.secondOption == '1'):
                    print('==================== Step describe feature ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].describe())
                elif(self.secondOption == '2'):
                    print('==================== Step feature value counts  ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].value_counts())
                elif(self.secondOption == '3'):
                    print('==================== Step feature unique values  ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].unique())
                elif(self.secondOption == '4'):
                    print('==================== Step remove outliers from feature ====================')
                    x = x[np.abs(zscore(x[input('Enter feature:') if self.isPrompt else self.entry.name])) < int(input('Enter abs z score') if self.isPrompt else self.entry.value)]
                elif(self.secondOption == '5'):
                    print('==================== Step unique feature values  ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].value_counts().head(int(input('Enter top value:') if self.isPrompt else self.entry.value)))
                elif(self.secondOption == '6'):
                    print('==================== Step combine feature ====================')
                    featureNum = input('Enter numerator feature:') if self.isPrompt else self.entry.name
                    featureDen = input('Enter denominator feature:') if self.isPrompt else self.entry.value
                    x[featureNum + '_per_' + featureDen] = x[featureNum] / x[featureDen]
                    self.dataset.dataframe = x
                elif(self.secondOption == '7'):
                    print('==================== Step correlation between features ====================')
                    corr = x.corr()
                    print(corr[input('Enter feature:') if self.isPrompt else self.entry.name].sort_values(ascending = False).head(64))
                elif(self.secondOption == '8'):
                    print('==================== Step ANOVA feature selection (numerical) ====================')
                    features_origin = input('Enter origin features separated by semicolon:') if self.isPrompt else self.entry.name
                    features_origin_split = features_origin.split(';')
                    features = input('Enter combined features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    features_split.extend(features_origin_split)
                    label = input('Enter label feature:') if self.isPrompt else self.extraEntry.name
                    df = pd.DataFrame(x, columns = features_split)
                    fs = SelectKBest(score_func = f_classif, k = int(input('Choose number features to be selected:') if self.isPrompt else self.extraEntry.value))
                    z = fs.fit_transform(df, x[label])
                    mask = fs.get_support()
                    new_features = x.columns[mask]
                    z = pd.DataFrame(x, columns = new_features)
                    columns = []                                 
                    for orig in features_origin_split:
                        if orig not in z:
                            columns.append(orig)
                    x = pd.DataFrame(x, columns = columns)
                    x = x.join(z)
                    self.dataset.dataframe = x
                elif(self.secondOption == '9'):
                    print('==================== Step Chi-square feature selection (categorical) ====================')
                    features_origin = input('Enter origin features separated by semicolon:') if self.isPrompt else self.entry.name
                    features_origin_split = features_origin.split(';')
                    features = input('Enter combined features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    features_split.extend(features_origin_split)
                    label = input('Enter label feature:') if self.isPrompt else self.extraEntry.name
                    df = pd.DataFrame(x, columns = features_split)
                    fs = SelectKBest(score_func = chi2, k = int(input('Choose number features to be selected:') if self.isPrompt else self.extraEntry.value))
                    z = fs.fit_transform(df, x[label])
                    mask = fs.get_support()
                    new_features = x.columns[mask]
                    z = pd.DataFrame(x, columns = new_features)
                    columns = []                                 
                    for orig in features_origin_split:
                        if orig not in z:
                            columns.append(orig)
                    x = pd.DataFrame(x, columns = columns)
                    x = x.join(z)
                    self.dataset.dataframe = x
                elif(self.secondOption == '10'):
                    print('==================== Step Mutual information feature selection (categorical) ====================')
                    features_origin = input('Enter origin features separated by semicolon:') if self.isPrompt else self.entry.name
                    features_origin_split = features_origin.split(';')
                    features = input('Enter numerical features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    features_split.extend(features_origin_split)
                    label = input('Enter label feature:') if self.isPrompt else self.extraEntry.name
                    df = pd.DataFrame(x, columns = features_split)
                    fs = SelectKBest(score_func = mutual_info_classif, k = int(input('Choose number features to be selected:') if self.isPrompt else self.extraEntry.value))
                    z = fs.fit_transform(df, x[label])
                    mask = fs.get_support()
                    new_features = x.columns[mask]
                    z = pd.DataFrame(x, columns = new_features)
                    columns = []                                 
                    for orig in features_origin_split:
                        if orig not in z:
                            columns.append(orig)
                    x = pd.DataFrame(x, columns = columns)
                    x = x.join(z)
                    self.dataset.dataframe = x
                elif(self.secondOption == '11'):
                    print('==================== Step encode feature (categorical) ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = OrdinalEncoder().fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '12'):
                    print('==================== Step encode feature one binary attribute (categorical) ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = OneHotEncoder().fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '13'):
                    print('==================== Step encode feature (ordinal) ====================')
                    scale_mapper = {}
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    if self.isPrompt:
                        for val in x[feature].unique():
                            num = input('Enter number for ' + val)
                            scale_mapper.update({val : int(num)})
                    else:
                        scale_mapper = self.entry.value
                    x[feature] = x[feature].replace(scale_mapper)
                    self.dataset.dataframe = x
                elif(self.secondOption == '14'):
                    print('==================== Step combine set of features ====================')
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.name
                    features_split = features.split(';')
                    features_split_perm = itertools.permutations(features_split, r = 2)
                    jointFeature = ["_per_".join(x) for x in features_split_perm]
                    for cal in jointFeature:
                        sep = cal.split('_per_')
                        x[cal] = x[sep[0]] / x[sep[1]]
                    self.dataset.dataframe = x   
                elif(self.secondOption == '15'):
                    print('==================== Step search feature instance ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    value = input('Enter feature value (only number):') if self.isPrompt else self.entry.value
                    if(self.isPrompt):
                        print('1 - Big than (>)')
                        print('2 - Big equal than (>=)')
                        print('3 - Less than (<)')
                        print('4 - Less equal than (<=)')
                        print('5 - Equal (==)')
                        print('6 - Not equal (!=)')
                    operation = input('Select operation number') if self.isPrompt else self.extraEntry.name
                    if(operation == '1'):
                        print(x[(x[feature] > float(value))])
                    elif(operation == '2'):
                        print(x[(x[feature] >= float(value))])
                    elif(operation == '3'):
                        print(x[(x[feature] < float(value))])
                    elif(operation == '4'):
                        print(x[(x[feature] <= float(value))])
                    elif(operation == '5'):
                        print(x[(x[feature] == float(value))])
                    elif(operation == '6'):
                        print(x[(x[feature] != float(value))])
                    self.dataset.dataframe = x
                elif(self.secondOption == '16'):
                    print('==================== Step show feature median ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].median())
                elif(self.secondOption == '17'):
                    print('==================== Step show feature mode ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].mode())
                elif(self.secondOption == '18'):
                    print('==================== Step show feature variance ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].var())
                elif(self.secondOption == '19'):
                    print('==================== Step show feature skewness ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].skew())
                elif(self.secondOption == '20'):
                    print('==================== Step show feature kurtosis ====================')
                    print(x[input('Enter feature:') if self.isPrompt else self.entry.name].kurt())
                elif(self.secondOption == '21'):
                    clear_output()
                elif(self.secondOption == '22'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
                
    def executeDuplicatesFlow(self, x):    
        while True:
            try:
                if(self.isPrompt):
                    self.printDuplicatesMenu()
                    self.secondOption = input('Choose option:')
                
                if(self.secondOption == '1'):
                    print('==================== Step any duplicates ====================')
                    print(x.duplicated().any())
                elif(self.secondOption == '2'):
                    print('==================== Step show duplicated values ====================')
                    print(x[x.duplicated()])
                elif(self.secondOption == '3'):
                    print('==================== Step remove duplicated values ====================')
                    x.drop_duplicates(inplace = True)
                elif(self.secondOption == '4'):
                    clear_output()
                elif(self.secondOption == '5'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
                
    def executeMissingValuesFlow(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMissingValuesMenu()
                    self.secondOption = input('Choose option:')
    
                if(self.secondOption == '1'):
                    print('==================== Step show total null values ====================')
                    print(x.isnull().sum())
                elif(self.secondOption == '2'):
                    print('==================== Step remove null values ====================')
                    x.dropna(inplace = True)
                    self.dataset.dataframe = x
                elif(self.secondOption == '3'):
                    print('==================== Step remove instance ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    value = input('Enter feature value (only number):') if self.isPrompt else self.entry.value
                    if(self.isPrompt):
                        print('1 - Big than (>)')
                        print('2 - Big equal than (>=)')
                        print('3 - Less than (<)')
                        print('4 - Less equal than (<=)')
                        print('5 - Equal (==)')
                        print('6 - Not equal (!=)')
                    operation = input('Select operation number') if self.isPrompt else self.extraEntry.name
                    if(operation == '1'):
                        x.drop(x[(x[feature] > float(value))].index, inplace = True)
                    elif(operation == '2'):
                        x.drop(x[(x[feature] >= float(value))].index, inplace = True)
                    elif(operation == '3'):
                        x.drop(x[(x[feature] < float(value))].index, inplace = True)
                    elif(operation == '4'):
                        x.drop(x[(x[feature] <= float(value))].index, inplace = True)
                    elif(operation == '5'):
                        x.drop(x[(x[feature] == float(value))].index, inplace = True)
                    elif(operation == '6'):
                        x.drop(x[(x[feature] != float(value))].index, inplace = True)
                    self.dataset.dataframe = x
                elif(self.secondOption == '4'):
                    print('==================== Step remove feature ====================')
                    x.drop(input('Enter feature:') if self.isPrompt else self.entry.name, inplace = True, axis = 1)
                    self.dataset.dataframe = x
                elif(self.secondOption == '5'):
                    print('==================== Step value imputation with mean ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = SimpleImputer(missing_values=0, strategy='mean').fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '6'):
                    print('==================== Step value imputation with median ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = SimpleImputer(missing_values=0, strategy='median').fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '7'):
                    print('==================== Step value imputation with most frequent ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '8'):
                    print('==================== Step value imputation with constant ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    if(str(x[feature].dtypes).startswith('int')):
                        constant = int(input('Enter constant:') if self.isPrompt else self.entry.value)
                    elif(str(x[feature].dtypes).startswith('float')):
                        constant = float(input('Enter constant:') if self.isPrompt else self.entry.value)
                    else:
                        constant = str(input('Enter constant:') if self.isPrompt else self.entry.value)
                    x[feature] = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = constant).fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '9'):
                    clear_output()
                elif(self.secondOption == '10'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
                
    def executeScalerFlow(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printScalerMenu()
                    self.secondOption = input('Choose option:')
                
                if(self.secondOption == '1'):
                    print('==================== Step min max scaler from feature ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = MinMaxScaler(feature_range=(0.01, 1)).fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '2'):
                    print('==================== Step standard scaler from feature ====================')
                    feature = input('Enter feature:') if self.isPrompt else self.entry.name
                    x[feature] = StandardScaler().fit_transform(x[feature].values.reshape(-1, 1))
                    self.dataset.dataframe = x
                elif(self.secondOption == '3'):
                    clear_output()
                elif(self.secondOption == '4'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())
                
class GraphOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        
    def printMenu(self):
        print('1 - Relation')
        print('2 - Correlation')
        print('3 - Heatmap')
        print('4 - Boxplot feature')
        print('5 - Boxplot all features')
        print('6 - Histogram')
        print('7 - Histogram all features')
        print('8 - Relation density distribution')
        print('9 - Jointplot')
        print('10 - Covariance')
        print('11 - Profiling')
        print('12 - Clear output')
        print('13 - Return main menu')
    
    def fit(self, x):
        return self
        
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')

                if(self.option == '1'):
                    print('==================== Step relation graph ====================')
                    sns.relplot(data = x, x = input('Enter first feature:') if self.isPrompt else self.entry.name, y = input('Enter second feature:') if self.isPrompt else self.entry.value)
                    plt.show()
                elif(self.option == '2'):
                    print('==================== Step correlation graph ====================')
                    print(x.corr())
                elif(self.option == '3'):
                    print('==================== Step heatmap graph ====================')
                    corr = x.corr()
                    sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns)
                    plt.show()
                elif(self.option == '4'): 
                    print('==================== Step boxplot feature graph ====================')
                    sns.boxplot(x = x[input('Enter feature:') if self.isPrompt else self.entry.name])
                    plt.show()
                elif(self.option == '5'):
                    print('==================== Step boxplot all features graph ====================')
                    for column in x.columns:
                        sns.boxplot(x = x[column])
                        plt.show()
                elif(self.option == '6'):
                    print('==================== Step histogram graph ====================')
                    sns.displot(x[input('Enter feature:') if self.isPrompt else self.entry.name], kde = True)
                    plt.show()
                elif(self.option == '7'):
                    print('==================== Step histogram all features graph ====================')
                    for column in x.columns:
                        sns.displot(x[column], kde = True)
                        plt.show()
                elif(self.option == '8'):
                    print('==================== Step relation density distribution graph ====================')
                    sns.displot(x, x = input('Enter first feature:') if self.isPrompt else self.entry.name, hue = input('Enter second feature:') if self.isPrompt else self.entry.value, stat = "density", common_norm=False)
                    plt.show()
                elif(self.option == '9'):
                    print('==================== Step joinplot graph ====================')
                    sns.jointplot(data = x, x = input('Enter first feature:') if self.isPrompt else self.entry.name, y = input('Enter second feature:'), hue = input('Enter hue feature:') if self.isPrompt else self.entry.value)
                    plt.show()
                elif(self.option == '10'):
                    print(x.cov())
                elif(self.option == '11'):
                    profile = pp.ProfileReport(x)
                    profile.to_file(input('Enter profile output path') if self.isPrompt else self.entry.name)
                elif(self.option == '12'):
                    clear_output()
                elif(self.option == '13'):
                    break
                    
                if(not self.isPrompt):
                    return x
            except:
                print('An error occurred', sys.exc_info())

class SupervisedLearningCrossValidationOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry, extraEntry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        self.extraEntry = extraEntry
    
    def printMenu(self):
        print('1 - Logistic Regression')
        print('2 - Support Vector Machine Linear')
        print('3 - Support Vector Machine Non Linear')
        print('4 - Decision Tree')
        print('5 - KNN')
        print('6 - Naive Bayes')
        print('7 - Random Forest')
        print('8 - Clear output')
        print('9 - Return main menu')
    
    def calculate_best_metrics(self, X, y, clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, C_dict, gamma_dict, neighbor_dict, estimator_dict, criterion_dict, balance_dict):  
        print('Test Max Accuracy = ' + str(accuracy_dict.get(max(accuracy_dict, key = accuracy_dict.get))) + ('; Random State = ' + str(random_state_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if random_state_dict != None else '') + ('; Split = ' + str(split_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if split_dict != None else '') + ('; Holdout Method = ' + str(method)) + ('; C = ' + str(C_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if gamma_dict != None else '') + ('; Neighbor = ' + str(neighbor_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if neighbor_dict != None else '') + ('; Estimator = ' + str(estimator_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        if(method == 'KFold'):
            cv = KFold(n_splits = split_dict.get(max(accuracy_dict, key = accuracy_dict.get)), shuffle = True, random_state = random_state_dict.get(max(accuracy_dict, key = accuracy_dict.get)))
        elif(method == 'StratifiedKFold'):
            cv = StratifiedKFold(n_splits = split_dict.get(max(accuracy_dict, key = accuracy_dict.get)), shuffle = True, random_state = random_state_dict.get(max(accuracy_dict, key = accuracy_dict.get)))
        elif(method == 'RepeatedStratifiedKFold'):
            cv = RepeatedStratifiedKFold(n_splits = split_dict.get(max(accuracy_dict, key = accuracy_dict.get)), n_repeats = 2, random_state = random_state_dict.get(max(accuracy_dict, key = accuracy_dict.get)))                   
        y_predict = cross_val_predict(estimator = clf, X = X, y = y, cv = cv)
        print(confusion_matrix(y, y_predict))
        print('========================================')    
        print('Test Max Precision = ' + str(precision_dict.get(max(precision_dict, key = precision_dict.get))) + ('; Random State = ' + str(random_state_dict.get(max(precision_dict, key = precision_dict.get))) if random_state_dict != None else '') + ('; Split = ' + str(split_dict.get(max(precision_dict, key = precision_dict.get))) if split_dict != None else '') + ('; Holdout Method = ' + str(method)) + ('; C = ' + str(C_dict.get(max(precision_dict, key = precision_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(precision_dict, key = precision_dict.get))) if gamma_dict != None else '') + ('; Neighbor = ' + str(neighbor_dict.get(max(precision_dict, key = precision_dict.get))) if neighbor_dict != None else '') + ('; Estimator = ' + str(estimator_dict.get(max(precision_dict, key = precision_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(precision_dict, key = precision_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(precision_dict, key = precision_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        if(method == 'KFold'):
            cv = KFold(n_splits = split_dict.get(max(precision_dict, key = precision_dict.get)), shuffle = True, random_state = random_state_dict.get(max(precision_dict, key = precision_dict.get)))
        elif(method == 'StratifiedKFold'):
            cv = StratifiedKFold(n_splits = split_dict.get(max(precision_dict, key = precision_dict.get)), shuffle = True, random_state = random_state_dict.get(max(precision_dict, key = precision_dict.get)))
        elif(method == 'RepeatedStratifiedKFold'):
            cv = RepeatedStratifiedKFold(n_splits = split_dict.get(max(precision_dict, key = precision_dict.get)), n_repeats = 2, random_state = random_state_dict.get(max(precision_dict, key = precision_dict.get)))                                                
        y_predict = cross_val_predict(estimator = clf, X = X, y = y, cv = cv)
        print(confusion_matrix(y, y_predict))
        print('========================================')
        print('Test Max Recall = ' + str(recall_dict.get(max(recall_dict, key=recall_dict.get))) + ('; Random State = ' + str(random_state_dict.get(max(recall_dict, key=recall_dict.get))) if random_state_dict != None else '') + ('; Split = ' + str(split_dict.get(max(recall_dict, key=recall_dict.get))) if split_dict != None else '') + ('; Holdout Method = ' + str(method)) + ('; C = ' + str(C_dict.get(max(recall_dict, key = recall_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(recall_dict, key = recall_dict.get))) if gamma_dict != None else '') + ('; Neighbor = ' + str(neighbor_dict.get(max(recall_dict, key = recall_dict.get))) if neighbor_dict != None else '') + ('; Estimator = ' + str(estimator_dict.get(max(recall_dict, key = recall_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(recall_dict, key = recall_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(recall_dict, key = recall_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        if(method == 'KFold'):
            cv = KFold(n_splits = split_dict.get(max(recall_dict, key = recall_dict.get)), shuffle = True, random_state = random_state_dict.get(max(recall_dict, key = recall_dict.get)))
        elif(method == 'StratifiedKFold'):
            cv = StratifiedKFold(n_splits = split_dict.get(max(recall_dict, key = recall_dict.get)), shuffle = True, random_state = random_state_dict.get(max(recall_dict, key = recall_dict.get)))
        elif(method == 'RepeatedStratifiedKFold'):
            cv = RepeatedStratifiedKFold(n_splits = split_dict.get(max(recall_dict, key = recall_dict.get)), n_repeats = 2, random_state = random_state_dict.get(max(recall_dict, key = recall_dict.get)))                  
        y_predict = cross_val_predict(estimator = clf, X = X, y = y, cv = cv)
        print(confusion_matrix(y, y_predict))
        print('========================================')
        print('Test Max F1-Score = ' + str(f1_score_dict.get(max(f1_score_dict, key=f1_score_dict.get))) + ('; Random State = ' + str(random_state_dict.get(max(f1_score_dict, key=f1_score_dict.get))) if random_state_dict != None else '') + ('; Split = ' + str(split_dict.get(max(f1_score_dict, key=f1_score_dict.get))) if split_dict != None else '') + ('; Holdout Method = ' + str(method)) + ('; C = ' + str(C_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if gamma_dict != None else '') + ('; Neighbor = ' + str(neighbor_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if neighbor_dict != None else '') + ('; Estimator = ' + str(estimator_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        if(method == 'KFold'):
            cv = KFold(n_splits = split_dict.get(max(f1_score_dict, key = f1_score_dict.get)), shuffle = True, random_state = random_state_dict.get(max(f1_score_dict, key = f1_score_dict.get)))
        elif(method == 'StratifiedKFold'):
            cv = StratifiedKFold(n_splits = split_dict.get(max(f1_score_dict, key = f1_score_dict.get)), shuffle = True, random_state = random_state_dict.get(max(f1_score_dict, key = f1_score_dict.get)))
        elif(method == 'RepeatedStratifiedKFold'):
            cv = RepeatedStratifiedKFold(n_splits = split_dict.get(max(f1_score_dict, key = f1_score_dict.get)), n_repeats = 2, random_state = random_state_dict.get(max(f1_score_dict, key = f1_score_dict.get)))                   
        y_predict = cross_val_predict(estimator = clf, X = X, y = y, cv = cv)
        print(confusion_matrix(y, y_predict))
        print('========================================')
        print('Test Max Roc Auc Score = ' + str(roc_auc_score_dict.get(max(roc_auc_score_dict, key=roc_auc_score_dict.get))) + ('; Random State = ' + str(random_state_dict.get(max(roc_auc_score_dict, key=roc_auc_score_dict.get))) if random_state_dict != None else '') + ('; Split = ' + str(split_dict.get(max(roc_auc_score_dict, key=roc_auc_score_dict.get))) if split_dict != None else '') + ('; Holdout Method = ' + str(method)) + ('; C = ' + str(C_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if gamma_dict != None else '') + ('; Neighbor = ' + str(neighbor_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if neighbor_dict != None else '') + ('; Estimator = ' + str(estimator_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        if(method == 'KFold'):
            cv = KFold(n_splits = split_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)), shuffle = True, random_state = random_state_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)))
        elif(method == 'StratifiedKFold'):
            cv = StratifiedKFold(n_splits = split_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)), shuffle = True, random_state = random_state_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)))
        elif(method == 'RepeatedStratifiedKFold'):
            cv = RepeatedStratifiedKFold(n_splits = split_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)), n_repeats = 2, random_state = random_state_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get)))                  
        y_predict = cross_val_predict(estimator = clf, X = X, y = y, cv = cv)
        print(confusion_matrix(y, y_predict))
        print('========================================')
        df.to_csv(fileName)
        
    def fit(self, x):
        return self
    
    def transform(self, x):                
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                
                if(self.option == '1'):
                    print('==================== Logistic Regression ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random State', 'Holdout Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Logistic_Regression_CV.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                    
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    log_reg = LogisticRegression(max_iter = 1000000)
                    
                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    i = 0
                    for split in range(19, 25):
                        for random_state in range(0, 4):
                            if(option == 1):
                                cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'KFold'
                            elif(option == 2):
                                cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'StratifiedKFold'
                            elif(option == 3):
                                cv = RepeatedStratifiedKFold(n_splits = split, n_repeats = 2, random_state = random_state)
                                method = 'RepeatedStratifiedKFold'

                            results = cross_validate(estimator = log_reg, X = X, y = y, cv = cv, scoring = scoring)

                            accuracy_dict[i] = np.mean(results['test_accuracy'])
                            precision_dict[i] = np.mean(results['test_precision'])
                            recall_dict[i] = np.mean(results['test_recall'])
                            f1_score_dict[i] = np.mean(results['test_f1_score'])
                            roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                            split_dict[i] = split
                            random_state_dict[i] = random_state

                            df.loc[i] = ['Logistic Regression', split, random_state, method, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                            i = i + 1

                    self.calculate_best_metrics(X, y, log_reg, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, None, None, None, None, None, None)
                    
                elif(self.option == '2'):
                    print('==================== Support Vector Machine Linear ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random State', 'Holdout Method', 'C', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'SVM_Linear_CV.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    C_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                            
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    C_range = [1]
                    
                    i = 0
                    for split in range(20, 25):
                        for random_state in range(0, 3):
                            for C in C_range:
                                if(option == 1):
                                    cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                    method = 'KFold'
                                elif(option == 2):
                                    cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                    method = 'StratifiedKFold'
                                elif(option == 3):
                                    cv = RepeatedStratifiedKFold(n_splits = split, n_repeats=2, random_state = random_state)
                                    method = 'RepeatedStratifiedKFold'

                                svm_clf = LinearSVC(C = C, class_weight = 'balanced', max_iter = 1000000)

                                results = cross_validate(estimator = svm_clf, X = X, y = y, cv = cv, scoring = scoring)

                                accuracy_dict[i] = np.mean(results['test_accuracy'])
                                precision_dict[i] = np.mean(results['test_precision'])
                                recall_dict[i] = np.mean(results['test_recall'])
                                f1_score_dict[i] = np.mean(results['test_f1_score'])
                                roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                                split_dict[i] = split
                                random_state_dict[i] = random_state
                                C_dict[i] = C

                                df.loc[i] = ['SVM Linear', split, random_state, method, C, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                i = i + 1

                    self.calculate_best_metrics(X, y, svm_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, C_dict, None, None, None, None, None)
                
                elif(self.option == '3'):
                    print('==================== Support Vector Machine Non Linear ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random State', 'Holdout Method', 'C', 'gamma', 'Kernel', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'SVM_Non_Linear_CV.csv'
                    kernel_trick_list = ['linear', 'rbf']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    C_dict = dict()
                    gamma_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                    
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    C_range = [1e2]
                    gamma_range = [1e1]
                    
                    i = 0
                    for split in range(20, 25):
                        for random_state in range(0, 3):
                            for C in C_range:
                                for gamma in gamma_range:
                                    for idx, kernel in enumerate(kernel_trick_list):
                                        if(option == 1):
                                            cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                            method = 'KFold'
                                        elif(option == 2):
                                            cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                            method = 'StratifiedKFold'
                                        elif(option == 3):
                                            method = 'RepeatedStratifiedKFold'
                                            cv = RepeatedStratifiedKFold(n_splits = split, n_repeats=2, random_state = random_state)

                                        svm_clf = SVC(kernel = kernel, C = C, gamma = gamma)

                                        results = cross_validate(estimator = svm_clf, X = X, y = y, cv = cv, scoring = scoring)

                                        accuracy_dict[i] = np.mean(results['test_accuracy'])
                                        precision_dict[i] = np.mean(results['test_precision'])
                                        recall_dict[i] = np.mean(results['test_recall'])
                                        f1_score_dict[i] = np.mean(results['test_f1_score'])
                                        roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                                        split_dict[i] = split
                                        random_state_dict[i] = random_state
                                        C_dict[i] = C
                                        gamma_dict[i] = gamma

                                        df.loc[i] = ['SVM Non Linear', split, random_state, method, C, gamma, kernel, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                        i = i + 1

                    self.calculate_best_metrics(X, y, svm_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, C_dict, gamma_dict, None, None, None, None)
                
                elif(self.option == '4'):
                    print('==================== Decision Tree ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random State', 'Holdout Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Decision_Tree_CV.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                            
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]

                    tree_clf = DecisionTreeClassifier()
                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    i = 0
                    for split in range(19, 25):
                        for random_state in range(0, 4):
                            if(option == 1):
                                cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'KFold'
                            elif(option == 2):
                                cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'StratifiedKFold'
                            elif(option == 3):
                                cv = RepeatedStratifiedKFold(n_splits = split, n_repeats = 2, random_state = random_state)
                                method = 'RepeatedStratifiedKFold'

                            results = cross_validate(estimator = tree_clf, X = X, y = y, cv = cv, scoring = scoring)

                            accuracy_dict[i] = np.mean(results['test_accuracy'])
                            precision_dict[i] = np.mean(results['test_precision'])
                            recall_dict[i] = np.mean(results['test_recall'])
                            f1_score_dict[i] = np.mean(results['test_f1_score'])
                            roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                            split_dict[i] = split
                            random_state_dict[i] = random_state

                            df.loc[i] = ['Decision Tree', split, random_state, method, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                            i = i + 1
                                
                    self.calculate_best_metrics(X, y, tree_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, None, None, None, None, None, None)
                
                elif(self.option == '5'):
                    print('==================== KNN ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random_State', 'Number Neighbors', 'Holdout Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'KNN_CV.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    neighbor_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                            
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]

                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    i = 0
                    for split in range(19, 25):
                        for random_state in range(0, 4):
                            for neighbor in range(1, 10):
                                knn_clf = KNeighborsClassifier(n_neighbors = neighbor)
                                if(option == 1):
                                    cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                    method = 'KFold'
                                elif(option == 2):
                                    cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                    method = 'StratifiedKFold'
                                elif(option == 3):
                                    cv = RepeatedStratifiedKFold(n_splits = split, n_repeats = 2, random_state = random_state)
                                    method = 'RepeatedStratifiedKFold'

                                results = cross_validate(estimator = knn_clf, X = X, y = y, cv = cv, scoring = scoring)

                                accuracy_dict[i] = np.mean(results['test_accuracy'])
                                precision_dict[i] = np.mean(results['test_precision'])
                                recall_dict[i] = np.mean(results['test_recall'])
                                f1_score_dict[i] = np.mean(results['test_f1_score'])
                                roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                                split_dict[i] = split
                                random_state_dict[i] = random_state
                                neighbor_dict[i] = neighbor

                                df.loc[i] = ['KNN', split, random_state, neighbor, method, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                i = i + 1
                                
                    self.calculate_best_metrics(X, y, knn_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, None, None, neighbor_dict, None, None, None)
                
                elif(self.option == '6'):
                    print('==================== Naive Bayes ====================')
                    
                    df = pd.DataFrame(columns=['Algorithm', 'Split', 'Random_State', 'Holdout Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Naive_Bayes_CV.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                            
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]

                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    i = 0
                    
                    gnb_clf = GaussianNB()
                    
                    for split in range(19, 25):
                        for random_state in range(0, 4):
                            if(option == 1):
                                cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'KFold'
                            elif(option == 2):
                                cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                method = 'StratifiedKFold'
                            elif(option == 3):
                                cv = RepeatedStratifiedKFold(n_splits = split, n_repeats = 2, random_state = random_state)
                                method = 'RepeatedStratifiedKFold'

                            results = cross_validate(estimator = gnb_clf, X = X, y = y, cv = cv, scoring = scoring)

                            accuracy_dict[i] = np.mean(results['test_accuracy'])
                            precision_dict[i] = np.mean(results['test_precision'])
                            recall_dict[i] = np.mean(results['test_recall'])
                            f1_score_dict[i] = np.mean(results['test_f1_score'])
                            roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                            split_dict[i] = split
                            random_state_dict[i] = random_state

                            df.loc[i] = ['Naive Bayes', split, random_state, method, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                            i = i + 1
                                
                    self.calculate_best_metrics(X, y, gnb_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, None, None, None, None, None, None)
                
                elif(self.option == '7'):
                    print('==================== Random Forest ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Split', 'Random_State', 'Holdout Method', 'Estimator', 'Criterion', 'Balance', 'Holdout Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Random_Forest_CV.csv'
                    criterion_list = ['gini', 'entropy']
                    balance_list = ['balanced', 'balanced_subsample']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    split_dict = dict()
                    random_state_dict = dict()
                    estimator_dict = dict()
                    criterion_dict = dict()
                    balance_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    while True:
                        if(self.isPrompt):
                            print('1 - KFold')
                            print('2 - StratifiedKFold')
                            print('3 - RepeatedStratifiedKFold')
                        option = input('Select Fold option:') if self.isPrompt else self.extraEntry.name
                        if(option > 0 and option < 4):
                            break
                            
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]

                    scoring = {'accuracy' : make_scorer(accuracy_score), 
                               'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 
                               'f1_score' : make_scorer(f1_score),
                               'roc_auc_score' : make_scorer(roc_auc_score)}
                    
                    i = 0
                    for split in range(19, 21):
                        for random_state in range(0, 2):
                            for estimator in range(10, 20, 10):
                                for idx1, criterion in enumerate(criterion_list):
                                    for idx2, balance in enumerate(balance_list):
                                        rf_clf = RandomForestClassifier(n_estimators = estimator, criterion = criterion, class_weight = balance)
                                        
                                        if(option == 1):
                                            cv = KFold(n_splits = split, shuffle = True, random_state = random_state)
                                            method = 'KFold'
                                        elif(option == 2):
                                            cv = StratifiedKFold(n_splits = split, shuffle = True, random_state = random_state)
                                            method = 'StratifiedKFold'
                                        elif(option == 3):
                                            cv = RepeatedStratifiedKFold(n_splits = split, n_repeats = 2, random_state = random_state)
                                            method = 'RepeatedStratifiedKFold'

                                        results = cross_validate(estimator = rf_clf, X = X, y = y, cv = cv, scoring = scoring)

                                        accuracy_dict[i] = np.mean(results['test_accuracy'])
                                        precision_dict[i] = np.mean(results['test_precision'])
                                        recall_dict[i] = np.mean(results['test_recall'])
                                        f1_score_dict[i] = np.mean(results['test_f1_score'])
                                        roc_auc_score_dict[i] = np.mean(results['test_roc_auc_score'])

                                        split_dict[i] = split
                                        random_state_dict[i] = random_state
                                        estimator_dict[i] = estimator
                                        criterion_dict[i] = criterion
                                        balance_dict[i] = balance

                                        df.loc[i] = ['Random Forest', split, random_state, method, estimator, criterion, balance, self.extraEntry.name, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                        i = i + 1
                                
                    self.calculate_best_metrics(X, y, rf_clf, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, random_state_dict, split_dict, method, None, None, None, estimator_dict, criterion_dict, balance_dict)
                
                elif(self.option == '8'):
                    clear_output()
                elif(self.option == '9'):
                    break
                    
                if(not self.isPrompt):
                    return x
                
            except:
                print('An error occurred', sys.exc_info())

class SupervisedLearningOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
    
    def printMenu(self):
        print('1 - Logistic Regression')
        print('2 - Support Vector Machine Linear')
        print('3 - Support Vector Machine Non Linear')
        print('4 - Decision Tree')
        print('5 - KNN')
        print('6 - Naive Bayes')
        print('7 - Random Forest')
        print('8 - Clear output')
        print('9 - Return main menu')
    
    def calculate_best_metrics(self, y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, C_dict, gamma_dict, neighbor_dict, estimator_dict, criterion_dict, balance_dict):  
        print('Test Max Accuracy = ' + str(accuracy_dict.get(max(accuracy_dict, key = accuracy_dict.get))) + '; Size = ' + str(size_dict.get(max(accuracy_dict, key = accuracy_dict.get))) + ('; C = ' + str(C_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if gamma_dict != None else '') + ('; Number Neighbor = ' + str(neighbor_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if neighbor_dict != None else '') + ('; Number Estimator = ' + str(estimator_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(accuracy_dict, key = accuracy_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print('========================================')    
        print('Test Max Precision = ' + str(precision_dict.get(max(precision_dict, key = precision_dict.get))) + '; Size = ' + str(size_dict.get(max(precision_dict, key = precision_dict.get))) + ('; C = ' + str(C_dict.get(max(precision_dict, key = precision_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(precision_dict, key = precision_dict.get))) if gamma_dict != None else '') + ('; Number Neighbor = ' + str(neighbor_dict.get(max(precision_dict, key = precision_dict.get))) if neighbor_dict != None else '')  + ('; Number Estimator = ' + str(estimator_dict.get(max(precision_dict, key = precision_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(precision_dict, key = precision_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(precision_dict, key = precision_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print('========================================')
        print('Test Max Recall = ' + str(recall_dict.get(max(recall_dict, key = recall_dict.get))) + '; Size = ' + str(size_dict.get(max(recall_dict, key = recall_dict.get))) + ('; C = ' + str(C_dict.get(max(recall_dict, key = recall_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(recall_dict, key = recall_dict.get))) if gamma_dict != None else '') + ('; Number Neighbor = ' + str(neighbor_dict.get(max(recall_dict, key = recall_dict.get))) if neighbor_dict != None else '') + ('; Number Estimator = ' + str(estimator_dict.get(max(recall_dict, key = recall_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(recall_dict, key = recall_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(recall_dict, key = recall_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print('========================================')
        print('Test Max F1-Score = ' + str(accuracy_dict.get(max(f1_score_dict, key = f1_score_dict.get))) + '; Size = ' + str(size_dict.get(max(f1_score_dict, key = f1_score_dict.get))) + ('; C = ' + str(C_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if gamma_dict != None else '') + ('; Number Neighbor = ' + str(neighbor_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if neighbor_dict != None else '') + ('; Number Estimator = ' + str(estimator_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(f1_score_dict, key = f1_score_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print('========================================')
        print('Test Max Roc Auc Score = ' + str(roc_auc_score_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) + '; Size = ' + str(size_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) + ('; C = ' + str(C_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if C_dict != None else '') + ('; Gamma = ' + str(gamma_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if gamma_dict != None else '') + ('; Number Neighbor = ' + str(neighbor_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if neighbor_dict != None else '') + ('; Number Estimator = ' + str(estimator_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if estimator_dict != None else '') + ('; Criterion = ' + str(criterion_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if criterion_dict != None else '') + ('; Balance = ' + str(balance_dict.get(max(roc_auc_score_dict, key = roc_auc_score_dict.get))) if balance_dict != None else ''))
        print('')
        print('Confusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print('========================================')
        df.to_csv(fileName)
        
    def fit(self, x):
        return self
    
    def transform(self, x):                
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                
                if(self.option == '1'):
                    print('==================== Logistic Regression ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Logistic_Regression.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    log_reg = LogisticRegression(max_iter = 1000000)
                    
                    i = 0
                    for size in np.linspace(0.2, 0.3, 3):
                    
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)
                        
                        log_reg.fit(X_train, y_train)
                        y_pred = log_reg.predict(X_test)

                        accuracy_dict[i] = accuracy_score(y_test, y_pred)
                        precision_dict[i] = precision_score(y_test, y_pred)
                        recall_dict[i] = recall_score(y_test, y_pred)
                        f1_score_dict[i] = f1_score(y_test, y_pred)
                        roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                        
                        size_dict[i] = size

                        df.loc[i] = ['Logistic Regression', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                        i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, None, None, None, None, None, None)
                
                elif(self.option == '2'):
                    print('==================== Support Vector Machine Linear ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'C', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'SVM_Linear.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    C_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    C_range = [1e-1, 1, 1e1]
                    
                    i = 0
                    for size in np.linspace(0.2, 0.3, 3):
                    
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                        for C in C_range:
                            svm_clf = LinearSVC(C = C, class_weight = 'balanced', max_iter = 1000000)

                            svm_clf.fit(X_train, y_train)
                            y_pred = svm_clf.predict(X_test)

                            accuracy_dict[i] = accuracy_score(y_test, y_pred)
                            precision_dict[i] = precision_score(y_test, y_pred)
                            recall_dict[i] = recall_score(y_test, y_pred)
                            f1_score_dict[i] = f1_score(y_test, y_pred)
                            roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                            size_dict[i] = size
                            C_dict[i] = C

                            df.loc[i] = ['SVM Linear', size, C, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                            i = i + 1
                                
                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, C_dict, None, None, None, None, None)
                
                elif(self.option == '3'):
                    print('==================== Support Vector Machine Non Linear ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'C', 'gamma', 'Kernel', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'SVM_Non_Linear.csv'
                    kernel_trick_list = ['linear', 'rbf']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    C_dict = dict()
                    gamma_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    C_range = [1e-2, 1, 1e2]
                    gamma_range = [1e-1, 1, 1e1]
                    
                    i = 0
                    for size in np.linspace(0.2, 0.3, 3):
                    
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                        for C in C_range:
                            for gamma in gamma_range:
                                for idx, kernel in enumerate(kernel_trick_list):
                                    svm_clf = SVC(kernel = kernel, C = C, gamma = gamma)

                                    svm_clf.fit(X_train, y_train)
                                    y_pred = svm_clf.predict(X_test)

                                    accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                    precision_dict[i] = precision_score(y_test, y_pred)
                                    recall_dict[i] = recall_score(y_test, y_pred)
                                    f1_score_dict[i] = f1_score(y_test, y_pred)
                                    roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                    size_dict[i] = size
                                    C_dict[i] = C
                                    gamma_dict[i] = gamma

                                    df.loc[i] = ['SVM Non Linear', size, C, gamma, kernel, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                    i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, C_dict, gamma_dict, None, None, None, None)
                
                elif(self.option == '4'):
                    print('==================== Decision Tree ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Decision_Tree.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]

                    tree_clf = DecisionTreeClassifier()
                    
                    i = 0
                    for size in np.linspace(0.2, 0.3, 3):
                    
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                        tree_clf.fit(X_train, y_train)
                        y_pred = tree_clf.predict(X_test)

                        accuracy_dict[i] = accuracy_score(y_test, y_pred)
                        precision_dict[i] = precision_score(y_test, y_pred)
                        recall_dict[i] = recall_score(y_test, y_pred)
                        f1_score_dict[i] = f1_score(y_test, y_pred)
                        roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                        
                        size_dict[i] = size

                        df.loc[i] = ['Decision Tree', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                        i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, None, None, None, None, None, None)
                
                elif(self.option == '5'):
                    print('==================== KNN ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'Number Neighbors', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'KNN.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    neighbor_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    for neighbor in range(1, 10):
                        knn_clf = KNeighborsClassifier(n_neighbors = neighbor)

                        for size in np.linspace(0.2, 0.3, 3):

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                            knn_clf.fit(X_train, y_train)
                            y_pred = knn_clf.predict(X_test)

                            accuracy_dict[i] = accuracy_score(y_test, y_pred)
                            precision_dict[i] = precision_score(y_test, y_pred)
                            recall_dict[i] = recall_score(y_test, y_pred)
                            f1_score_dict[i] = f1_score(y_test, y_pred)
                            roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                            
                            size_dict[i] = size
                            neighbor_dict[i] = neighbor

                            df.loc[i] = ['KNN', size, neighbor, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                            i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, None, None, neighbor_dict, None, None, None)
                
                elif(self.option == '6'):
                    print('==================== Naive Bayes ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Naive_Bayes.csv'
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                   
                    gnb_clf = GaussianNB()

                    for size in np.linspace(0.2, 0.3, 3):

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                        gnb_clf.fit(X_train, y_train)
                        y_pred = gnb_clf.predict(X_test)

                        accuracy_dict[i] = accuracy_score(y_test, y_pred)
                        precision_dict[i] = precision_score(y_test, y_pred)
                        recall_dict[i] = recall_score(y_test, y_pred)
                        f1_score_dict[i] = f1_score(y_test, y_pred)
                        roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                        
                        size_dict[i] = size

                        df.loc[i] = ['Naive Bayes', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                        i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, None, None, None, None, None, None)
                
                elif(self.option == '7'):
                    print('==================== Random Forest ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Test Size', 'Number Estimator', 'Criterion', 'Balance', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'Random_Forest.csv'
                    criterion_list = ['gini', 'entropy']
                    balance_list = ['balanced', 'balanced_subsample']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    size_dict = dict()
                    estimator_dict = dict()
                    criterion_dict = dict()
                    balance_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                   
                    for estimator in range(10, 120, 30):
                        for idx1, criterion in enumerate(criterion_list):
                            for idx2, balance in enumerate(balance_list):
                                rf_clf = RandomForestClassifier(n_estimators = estimator, criterion = criterion, class_weight = balance)

                                for size in np.linspace(0.2, 0.3, 3):

                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                                    rf_clf.fit(X_train, y_train)
                                    y_pred = rf_clf.predict(X_test)

                                    accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                    precision_dict[i] = precision_score(y_test, y_pred)
                                    recall_dict[i] = recall_score(y_test, y_pred)
                                    f1_score_dict[i] = f1_score(y_test, y_pred)
                                    roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                    size_dict[i] = size
                                    estimator_dict[i] = estimator
                                    criterion_dict[i] = criterion
                                    balance_dict[i] = balance

                                    df.loc[i] = ['Random Forest', size, estimator, criterion, balance, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]
                                    i = i + 1

                    self.calculate_best_metrics(y_test, y_pred, df, fileName, accuracy_dict, precision_dict, recall_dict, f1_score_dict, roc_auc_score_dict, size_dict, None, None, None, estimator_dict, criterion_dict, balance_dict)
                
                elif(self.option == '8'):
                    clear_output()
                elif(self.option == '9'):
                    break
                    
                if(not self.isPrompt):
                    return x
                
            except:
                print('An error occurred', sys.exc_info())

class UnsupervisedLearningOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        
    def printMenu(self):
        print('1 - KMeans')
        print('2 - Spectral Clustering')
        print('3 - Mean Shift')
        print('4 - DBSCAN')
        print('5 - Birch')
        print('6 - Clear output')
        print('7 - Return main menu')
        
    def show_metrics(self, df, fileName):
        df.to_csv(fileName)
        
    def fit(self, x):
        return self
    
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                
                if(self.option == '1'):
                    print('==================== KMeans ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Type', 'Max Iterations', 'Number Cluster', 'Test Size', 'N_Init', 'Silhouette', 'Homogeneity', 'Completeness', 'V Measure', 'Adjusted Rand', 'Adjusted Mutual Info'])
                    fileName = 'KMeans.csv'
                    type_list = ['train', 'test']
                    
                    silhouette_score_dict = dict()
                    homogeneity_score_dict = dict()
                    completeness_score_dict = dict()
                    v_measure_score_dict = dict()
                    adjusted_rand_score_dict = dict()
                    adjusted_mutual_info_score_dict = dict()
                    max_iter_dict = dict()
                    n_cluster_dict = dict()
                    size_dict = dict()
                    n_init_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    for idx1, _type in enumerate(type_list):
                        for max_iter in np.arange(1, 602, 60):
                            for n_cluster in np.arange(2, 6):
                                for size in np.linspace(0.2, 0.3, 3):
                                    for n_init in np.arange(1, 5):

                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                                        kmeans = KMeans(n_clusters = n_cluster, max_iter = max_iter, random_state = 1, n_init = n_init)

                                        kmeans.fit(X_train if _type == 'train' else X_test)
                                        y_pred = kmeans.predict(X_train if _type == 'train' else X_test)

                                        silhouette_score_dict[i] = silhouette_score(X_train if _type == 'train' else X_test, y_pred)
                                        homogeneity_score_dict[i] = homogeneity_score(y_train if _type == 'train' else y_test, y_pred)
                                        completeness_score_dict[i] = completeness_score(y_train if _type == 'train' else y_test, y_pred)
                                        v_measure_score_dict[i] = v_measure_score(y_train if _type == 'train' else y_test, y_pred)
                                        adjusted_rand_score_dict[i] = adjusted_rand_score(y_train if _type == 'train' else y_test, y_pred)
                                        adjusted_mutual_info_score_dict[i] = adjusted_mutual_info_score(y_train if _type == 'train' else y_test, y_pred)

                                        max_iter_dict[i] = max_iter
                                        n_cluster_dict[i] = n_cluster
                                        size_dict[i] = size
                                        n_init_dict[i] = n_init

                                        df.loc[i] = ['KMeans', _type, max_iter, n_cluster, size, n_init, silhouette_score_dict[i], homogeneity_score_dict[i], completeness_score_dict[i], v_measure_score_dict[i], adjusted_rand_score_dict[i], adjusted_mutual_info_score_dict[i]]

                                        i = i + 1

                    self.show_metrics(df, fileName)
                
                elif(self.option == '2'):
                    print('==================== Spectral Clustering ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Number Cluster', 'N_Init', 'Gamma', 'Assign Label', 'Silhouette', 'Homogeneity', 'Completeness', 'V Measure', 'Adjusted Rand', 'Adjusted Mutual Info'])
                    fileName = 'Spectral_Clustering.csv'
                    assign_labels_list = ['kmeans', 'discretize']
                    
                    silhouette_score_dict = dict()
                    homogeneity_score_dict = dict()
                    completeness_score_dict = dict()
                    v_measure_score_dict = dict()
                    adjusted_rand_score_dict = dict()
                    adjusted_mutual_info_score_dict = dict()
                    n_cluster_dict = dict()
                    n_init_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    gamma_range = [1e-2, 1e-1, 1]
                    
                    i = 0
                    for n_cluster in np.arange(2, 4):
                        for n_init in np.arange(1, 3):
                            for gamma in gamma_range:
                                for idx1, assign_label in enumerate(assign_labels_list):

                                    clustering = SpectralClustering(n_clusters = n_cluster, assign_labels = assign_label, random_state = 1, n_init = n_init, gamma = gamma)

                                    y_pred = clustering.fit_predict(X)

                                    silhouette_score_dict[i] = silhouette_score(X, y_pred)
                                    homogeneity_score_dict[i] = homogeneity_score(y, y_pred)
                                    completeness_score_dict[i] = completeness_score(y, y_pred)
                                    v_measure_score_dict[i] = v_measure_score(y, y_pred)
                                    adjusted_rand_score_dict[i] = adjusted_rand_score(y, y_pred)
                                    adjusted_mutual_info_score_dict[i] = adjusted_mutual_info_score(y, y_pred)

                                    n_cluster_dict[i] = n_cluster
                                    n_init_dict[i] = n_init

                                    df.loc[i] = ['Spectral_Clustering', n_cluster, n_init, gamma, assign_label, silhouette_score_dict[i], homogeneity_score_dict[i], completeness_score_dict[i], v_measure_score_dict[i], adjusted_rand_score_dict[i], adjusted_mutual_info_score_dict[i]]

                                    i = i + 1
                                        
                    self.show_metrics(df, fileName)
                    
                if(self.option == '3'):
                    print('==================== Mean Shift ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Type', 'Max Iterations', 'Test Size', 'Bandwidth', 'Clusters', 'Silhouette', 'Homogeneity', 'Completeness', 'V Measure', 'Adjusted Rand', 'Adjusted Mutual Info'])
                    fileName = 'Mean_Shift.csv'
                    type_list = ['train', 'test']
                    
                    silhouette_score_dict = dict()
                    homogeneity_score_dict = dict()
                    completeness_score_dict = dict()
                    v_measure_score_dict = dict()
                    adjusted_rand_score_dict = dict()
                    adjusted_mutual_info_score_dict = dict()
                    max_iter_dict = dict()
                    bandwidth_dict = dict()
                    size_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    for idx1, _type in enumerate(type_list):
                        for max_iter in np.arange(200, 601, 100):
                            for bandwidth in np.arange(2, 6):
                                for size in np.linspace(0.2, 0.3, 3):

                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                                        mean_shift = MeanShift(max_iter = max_iter, bandwidth = bandwidth, bin_seeding = True)

                                        mean_shift.fit(X_train if _type == 'train' else X_test)
                                        y_pred = mean_shift.predict(X_train if _type == 'train' else X_test)
                                        
                                        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

                                        try:
                                            silhouette_score_dict[i] = silhouette_score(X_train if _type == 'train' else X_test, y_pred)
                                        except:
                                            silhouette_score_dict[i] = None
                                        try:
                                            homogeneity_score_dict[i] = homogeneity_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            homogeneity_score_dict[i] = None
                                        try:    
                                            completeness_score_dict[i] = completeness_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            completeness_score_dict[i] = None
                                        try:
                                            v_measure_score_dict[i] = v_measure_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            v_measure_score_dict[i] = None
                                        try:
                                            adjusted_rand_score_dict[i] = adjusted_rand_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            adjusted_rand_score_dict[i]
                                        try:
                                            adjusted_mutual_info_score_dict[i] = adjusted_mutual_info_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            adjusted_mutual_info_score_dict[i] = None
                                            
                                        max_iter_dict[i] = max_iter
                                        bandwidth_dict[i] = bandwidth
                                        size_dict[i] = size

                                        df.loc[i] = ['Mean Shift', _type, max_iter, size, bandwidth, n_clusters, silhouette_score_dict[i], homogeneity_score_dict[i], completeness_score_dict[i], v_measure_score_dict[i], adjusted_rand_score_dict[i], adjusted_mutual_info_score_dict[i]]

                                        i = i + 1

                    self.show_metrics(df, fileName)
                    
                elif(self.option == '4'):
                    print('==================== DBSCAN ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'EPS', 'Min Sample', 'Clusters', 'Silhouette', 'Homogeneity', 'Completeness', 'V Measure', 'Adjusted Rand', 'Adjusted Mutual Info'])
                    fileName = 'DBSCAN.csv'
                    
                    silhouette_score_dict = dict()
                    homogeneity_score_dict = dict()
                    completeness_score_dict = dict()
                    v_measure_score_dict = dict()
                    adjusted_rand_score_dict = dict()
                    adjusted_mutual_info_score_dict = dict()
                    eps_dict = dict()
                    min_sample_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    for eps in np.linspace(0.5, 4, 10):
                        for min_sample in np.arange(1, 5):
                            
                                dbscan = DBSCAN(eps = eps, min_samples = min_sample)

                                y_pred = dbscan.fit_predict(X)
                                
                                n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

                                try:
                                    silhouette_score_dict[i] = silhouette_score(X, y_pred)
                                except:
                                    silhouette_score_dict[i] = None
                                try:
                                    homogeneity_score_dict[i] = homogeneity_score(y, y_pred)
                                except:
                                    homogeneity_score_dict[i] = None
                                try:
                                    completeness_score_dict[i] = completeness_score(y, y_pred)
                                except:
                                    completeness_score_dict[i] = None
                                try:
                                    v_measure_score_dict[i] = v_measure_score(y, y_pred)
                                except:
                                    v_measure_score_dict[i] = None
                                try:
                                    adjusted_rand_score_dict[i] = adjusted_rand_score(y, y_pred)
                                except:
                                    adjusted_rand_score_dict[i] = None
                                try:
                                    adjusted_mutual_info_score_dict[i] = adjusted_mutual_info_score(y, y_pred)
                                except:
                                    adjusted_mutual_info_score_dict[i] = None

                                eps_dict[i] = eps
                                min_sample_dict[i] = min_sample

                                df.loc[i] = ['DBSCAN', eps, min_sample, n_clusters, silhouette_score_dict[i], homogeneity_score_dict[i], completeness_score_dict[i], v_measure_score_dict[i], adjusted_rand_score_dict[i], adjusted_mutual_info_score_dict[i]]

                                i = i + 1
                                        
                    self.show_metrics(df, fileName)
                
                if(self.option == '5'):
                    print('==================== Birch ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Type', 'Clusters', 'Size', 'Threshold', 'Branching Factor', 'Silhouette', 'Homogeneity', 'Completeness', 'V Measure', 'Adjusted Rand', 'Adjusted Mutual Info'])
                    fileName = 'Birch.csv'
                    type_list = ['train', 'test']
                    cluster_list = [None, 1, 2, 3]
                    
                    silhouette_score_dict = dict()
                    homogeneity_score_dict = dict()
                    completeness_score_dict = dict()
                    v_measure_score_dict = dict()
                    adjusted_rand_score_dict = dict()
                    adjusted_mutual_info_score_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    for idx1, _type in enumerate(type_list):
                        for idx2, cluster in enumerate(cluster_list):
                            for threshold in np.linspace(0.1, 2, 5):
                                for branching_factor in np.arange(20, 61, 20):
                                    for size in np.linspace(0.2, 0.3, 3):

                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)

                                        birch = Birch(n_clusters = cluster, threshold = threshold, branching_factor = branching_factor)

                                        birch.fit(X_train if _type == 'train' else X_test)
                                        y_pred = birch.predict(X_train if _type == 'train' else X_test)

                                        try:
                                            silhouette_score_dict[i] = silhouette_score(X_train if _type == 'train' else X_test, y_pred)
                                        except:
                                            silhouette_score_dict[i] = None
                                        try:
                                            homogeneity_score_dict[i] = homogeneity_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            homogeneity_score_dict[i] = None
                                        try:    
                                            completeness_score_dict[i] = completeness_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            completeness_score_dict[i] = None
                                        try:
                                            v_measure_score_dict[i] = v_measure_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            v_measure_score_dict[i] = None
                                        try:
                                            adjusted_rand_score_dict[i] = adjusted_rand_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            adjusted_rand_score_dict[i]
                                        try:
                                            adjusted_mutual_info_score_dict[i] = adjusted_mutual_info_score(y_train if _type == 'train' else y_test, y_pred)
                                        except:
                                            adjusted_mutual_info_score_dict[i] = None

                                        df.loc[i] = ['Birch', _type, cluster, size, threshold, branching_factor, silhouette_score_dict[i], homogeneity_score_dict[i], completeness_score_dict[i], v_measure_score_dict[i], adjusted_rand_score_dict[i], adjusted_mutual_info_score_dict[i]]

                                        i = i + 1

                    self.show_metrics(df, fileName)
                
                elif(self.option == '6'):
                    clear_output()
                elif(self.option == '7'):
                    break
                    
                if(not self.isPrompt):
                    return x
 
            except:
                print('An error occurred', sys.exc_info())
            
class UnsupervisedLearningPreProcessingOperation(Operation, BaseEstimator, TransformerMixin):
    def __init__(self, dataset, isPrompt, option, entry):
        Operation.__init__(self, dataset)
        self.isPrompt = isPrompt
        self.option = option
        self.entry = entry
        
    def printMenu(self):
        print('1 - Preprocessing Logistic Regression')
        print('2 - Preprocessing Naive Bayes')
        print('3 - Preprocessing Decision Tree')
        print('4 - Clear output')
        print('5 - Return main menu')
        
        
    def show_metrics(self, df, fileName):
        df.to_csv(fileName)
        
    def fit(self, x):
        return self
    
    def transform(self, x):
        while True:
            try:
                if(self.isPrompt):
                    self.printMenu()
                    self.option = input('Choose option:')
                
                if(self.option == '1'):
                    print('==================== Preprocessing Logistic Regression ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Preprocessing Algorithm', 'Best parameters', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'PreProcessing_Logistic_Regression.csv'
                    preprocessing_algorithm_list = ['none', 'kmeans', 'birch']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    
                    for idx, preprocessing_algorithm in enumerate(preprocessing_algorithm_list):
                        for size in np.linspace(0.2, 0.3, 3):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)
                            
                            if(preprocessing_algorithm == 'none'):
                                log_reg = LogisticRegression(max_iter = 1000000)
                                log_reg.fit(X_train, y_train)
                                y_pred = log_reg.predict(X_test)
                                
                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                                
                                df.loc[i] = ['Logistic regression', preprocessing_algorithm, '', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                            elif(preprocessing_algorithm == 'kmeans'):
                                pipeline = Pipeline([("kmeans", KMeans()), ("log_reg", LogisticRegression(max_iter = 1000000))])

                                param_grid = dict(kmeans__n_clusters = range(2, 6), kmeans__max_iter = np.arange(1, 602, 60))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Logistic regression', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1
                                
                            elif(preprocessing_algorithm == 'birch'):
                                pipeline = Pipeline([("birch", Birch()), ("log_reg", LogisticRegression(max_iter = 1000000))])

                                param_grid = dict(birch__n_clusters = [None, 1, 2, 3], birch__threshold = np.linspace(0.1, 2, 5), birch__branching_factor = np.arange(20, 61, 20))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Logistic regression', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                    self.show_metrics(df, fileName)
                
                elif(self.option == '2'):
                    print('==================== Preprocessing Naive Bayes ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Preprocessing Algorithm', 'Best parameters', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'PreProcessing_Naive_Bayes.csv'
                    preprocessing_algorithm_list = ['none', 'kmeans', 'birch']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    
                    for idx, preprocessing_algorithm in enumerate(preprocessing_algorithm_list):
                        for size in np.linspace(0.2, 0.3, 3):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)
                            
                            if(preprocessing_algorithm == 'none'):
                                gnb_clf = GaussianNB()
                                gnb_clf.fit(X_train, y_train)
                                y_pred = gnb_clf.predict(X_test)
                                
                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                                
                                df.loc[i] = ['Naive Bayes', preprocessing_algorithm, '', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                            elif(preprocessing_algorithm == 'kmeans'):
                                pipeline = Pipeline([("kmeans", KMeans()), ("naive_bayes", GaussianNB())])

                                param_grid = dict(kmeans__n_clusters = range(2, 6), kmeans__max_iter = np.arange(1, 602, 60))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3, verbose = 2)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Naive Bayes', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1
                                
                            elif(preprocessing_algorithm == 'birch'):
                                pipeline = Pipeline([("birch", Birch()), ("naive_bayes", GaussianNB())])

                                param_grid = dict(birch__n_clusters = [None, 1, 2, 3], birch__threshold = np.linspace(0.1, 2, 5), birch__branching_factor = np.arange(20, 61, 20))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3, verbose = 2)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Naive Bayes', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                    self.show_metrics(df, fileName)
                
                elif(self.option == '3'):
                    print('==================== Preprocessing Decision Tree ====================')
                    
                    df = pd.DataFrame(columns = ['Algorithm', 'Preprocessing Algorithm', 'Best parameters', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Roc-Auc-Score'])
                    fileName = 'PreProcessing_Decision_Tree.csv'
                    preprocessing_algorithm_list = ['none', 'kmeans', 'birch']
                    
                    accuracy_dict = dict()
                    precision_dict = dict()
                    recall_dict = dict()
                    f1_score_dict = dict()
                    roc_auc_score_dict = dict()
                    
                    label = input('Enter label feature:') if self.isPrompt else self.entry.name
                    features = input('Enter features separated by semicolon:') if self.isPrompt else self.entry.value
                    features_split = features.split(';')
                    X = pd.DataFrame(x, columns = features_split)
                    y = x[label]
                    
                    i = 0
                    
                    for idx, preprocessing_algorithm in enumerate(preprocessing_algorithm_list):
                        for size in np.linspace(0.2, 0.3, 3):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 0)
                            
                            if(preprocessing_algorithm == 'none'):
                                gnb_clf = DecisionTreeClassifier()
                                gnb_clf.fit(X_train, y_train)
                                y_pred = gnb_clf.predict(X_test)
                                
                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)
                                
                                df.loc[i] = ['Decision Tree', preprocessing_algorithm, '', size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                            elif(preprocessing_algorithm == 'kmeans'):
                                pipeline = Pipeline([("kmeans", KMeans()), ("decision_tree", DecisionTreeClassifier())])

                                param_grid = dict(kmeans__n_clusters = range(2, 6), kmeans__max_iter = np.arange(1, 602, 60))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Decision Tree', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1
                                
                            elif(preprocessing_algorithm == 'birch'):
                                pipeline = Pipeline([("birch", Birch()), ("decision_tree", DecisionTreeClassifier())])

                                param_grid = dict(birch__n_clusters = [None, 1, 2, 3], birch__threshold = np.linspace(0.1, 2, 5), birch__branching_factor = np.arange(20, 61, 20))
                                grid_clf = GridSearchCV(pipeline, param_grid, cv = 3)
                                grid_clf.fit(X_train, y_train)

                                y_pred = grid_clf.predict(X_test)

                                accuracy_dict[i] = accuracy_score(y_test, y_pred)
                                precision_dict[i] = precision_score(y_test, y_pred)
                                recall_dict[i] = recall_score(y_test, y_pred)
                                f1_score_dict[i] = f1_score(y_test, y_pred)
                                roc_auc_score_dict[i] = roc_auc_score(y_test, y_pred)

                                df.loc[i] = ['Decision_Tree', preprocessing_algorithm, grid_clf.best_params_, size, accuracy_dict[i], precision_dict[i], recall_dict[i], f1_score_dict[i], roc_auc_score_dict[i]]

                                i = i + 1

                    self.show_metrics(df, fileName)
                
                elif(self.option == '4'):
                    clear_output()
                elif(self.option == '5'):
                    break
                    
                if(not self.isPrompt):
                    return x
            
            except:
                print('An error occurred', sys.exc_info())

class PipelineOperation(Operation):
    def __init__(self, dataset):
        Operation.__init__(self, dataset)
    
    def transform(self):
        num_pipeline = Pipeline([('read_dataset', self.dataset), 
                                 ('show_columns_first', MetadataOperation(self.dataset, isPrompt = False, option = '1', entry = Entry(None, None))),
                                 ('lower_columns', MetadataOperation(self.dataset, isPrompt = False, option = '3', entry = Entry(None, None))),
                                 #('boxplot_all_first', GraphOperation(self.dataset, isPrompt = False, option = '5', entry = Entry(None, None))),
                                 ('show_columns_second', MetadataOperation(self.dataset, isPrompt = False, option = '1', entry = Entry(None, None))), 
                                 #('rename_column', MetadataOperation(self.dataset, isPrompt = False, option = '4', entry = Entry('average_montly_hours', 'average_monthly_hours'))),
                                 #('show_columns_third', MetadataOperation(self.dataset, isPrompt = False, option = '1', entry = Entry(None, None))),
                                 ('check_duplicates_first', DataOperation(self.dataset, isPrompt = False, firstOption = '3', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('show_shape_first', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '4', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('remove_duplicates', DataOperation(self.dataset, isPrompt = False, firstOption = '3', secondOption = '3', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('check_duplicates_second', DataOperation(self.dataset, isPrompt = False, firstOption = '3', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('describe_dataset_first', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('replace_glucose_mean', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '5', entry = Entry('glucose', None), extraEntry = Entry(None, None))),
                                 ('replace_skinthickness_mean', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '5', entry = Entry('skinthickness', None), extraEntry = Entry(None, None))),
                                 ('replace_bloodpressure_mean', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '5', entry = Entry('bloodpressure', None), extraEntry = Entry(None, None))),
                                 ('replace_bmi_median', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '6', entry = Entry('bmi', None), extraEntry = Entry(None, None))),
                                 ('replace_insulin_median', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '6', entry = Entry('insulin', None), extraEntry = Entry(None, None))),
                                 ('describe_dataset_second', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('boxplot_all_second', GraphOperation(self.dataset, isPrompt = False, option = '5', entry = Entry(None, None))),
                                 #('show_shape_second', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '4', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('show_type_columns', MetadataOperation(self.dataset, isPrompt = False, option = '2', entry = Entry(None, None))),
                                 #('show_values_department_first', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('department', None), extraEntry = Entry(None, None))),
                                 #('show_values_salary', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('salary', None), extraEntry = Entry(None, None))),
                                 #('encode_feature_department', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '13', entry = Entry('department', {'sales': 1, 'technical': 2, 'support': 3, 'IT': 4, 'RandD': 5, 'product_mng': 6, 'marketing': 7, 'accounting': 8, 'hr': 9, 'management': 10}), extraEntry = Entry(None, None))),
                                 #('show_values_department_second', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('department', None), extraEntry = Entry(None, None))),
                                 #('encode_feature_salary', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '13', entry = Entry('salary', {'low': 1, 'medium': 2, 'high': 3}), extraEntry = Entry(None, None))),
                                 #('encode_feature_left', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '13', entry = Entry('left', {0: -1}), extraEntry = Entry(None, None))),
                                 #('show_values_salary_second', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('salary', None), extraEntry = Entry(None, None))),
                                 #('describe_dataset_first', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_pregnancies', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('pregnancies', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_glucose', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('glucose', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_bloodpressure', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('bloodpressure', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_skinthickness', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('skinthickness', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_insulin', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('insulin', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_bmi', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('bmi', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_diabetespedigreefunction', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('diabetespedigreefunction', None), extraEntry = Entry(None, None))),
                                 ('min_max_scaler_age', DataOperation(self.dataset, isPrompt = False, firstOption = '5', secondOption = '1', entry = Entry('age', None), extraEntry = Entry(None, None))),
                                 #('describe_dataset_third', DataOperation(self.dataset, isPrompt = False, firstOption = '1', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('show_values_average_monthly_hours', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('average_monthly_hours', None), extraEntry = Entry(None, None))),
                                 #('relation_all_first', GraphOperation(self.dataset, isPrompt = False, option = '2', entry = Entry(None, None))),
                                 #('histogram_all', GraphOperation(self.dataset, isPrompt = False, option = '7', entry = Entry(None, None))),
                                 #('boxplot_all', GraphOperation(self.dataset, isPrompt = False, option = '5', entry = Entry(None, None))),
                                 #('check_null_first', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('remove_null', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '2', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 #('check_null_second', DataOperation(self.dataset, isPrompt = False, firstOption = '4', secondOption = '1', entry = Entry(None, None), extraEntry = Entry(None, None))),
                                 ('combine_features', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '14', entry = Entry('glucose;skinthickness;diabetespedigreefunction;insulin;age;pregnancies;bmi;bloodpressure', None), extraEntry = Entry(None, None))),
                                 #('show_columns_fourth', MetadataOperation(self.dataset, isPrompt = False, option = '1', entry = Entry(None, None))), 
                                 ('show_corr_first', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '7', entry = Entry('outcome', None), extraEntry = Entry(None, None))),
                                 #('anova', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '8', entry = Entry('pregnancies;glucose;bloodpressure;skinthickness;insulin;bmi;age;diabetespedigreefunction;outcome', 'satisfaction_level_per_last_evaluation;satisfaction_level_per_average_monthly_hours;satisfaction_level_per_time_spend_company;last_evaluation_per_satisfaction_level;last_evaluation_per_average_monthly_hours;last_evaluation_per_time_spend_company;average_monthly_hours_per_satisfaction_level;average_monthly_hours_per_last_evaluation;average_monthly_hours_per_time_spend_company;time_spend_company_per_satisfaction_level;time_spend_company_per_last_evaluation;time_spend_company_per_average_monthly_hours'), extraEntry = Entry('left', '10'))),
                                 #('anova', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '8', entry = Entry('pregnancies;glucose;bloodpressure;skinthickness;insulin;bmi;age;diabetespedigreefunction;outcome', 'pregnancies_per_glucose;pregnancies_per_bloodpressure;pregnancies_per_skinthickness;pregnancies_per_insulin;pregnancies_per_bmi;pregnancies_per_age;pregnancies_per_diabetespedigreefunction;glucose_per_pregnancies;glucose_per_bloodpressure;glucose_per_skinthickness;glucose_per_insulin;glucose_per_bmi;glucose_per_age;glucose_per_diabetespedigreefunction;bloodpressure_per_pregnancies;bloodpressure_per_glucose;bloodpressure_per_skinthickness;bloodpressure_per_insulin;bloodpressure_per_bmi;bloodpressure_per_age;bloodpressure_per_diabetespedigreefunction;skinthickness_per_pregnancies;skinthickness_per_glucose;skinthickness_per_bloodpressure;skinthickness_per_insulin;skinthickness_per_bmi;skinthickness_per_age;skinthickness_per_diabetespedigreefunction;insulin_per_pregnancies;insulin_per_glucose;insulin_per_bloodpressure;insulin_per_skinthickness;insulin_per_bmi;insulin_per_age;insulin_per_diabetespedigreefunction;age_per_pregnancies;age_per_glucose;age_per_bloodpressure;age_per_skinthickness;age_per_bmi;age_per_insulin;age_per_diabetespedigreefunction;diabetespedigreefunction_per_pregnancies;diabetespedigreefunction_per_glucose;diabetespedigreefunction_per_bloodpressure;diabetespedigreefunction_per_skinthickness;diabetespedigreefunction_per_bmi;diabetespedigreefunction_per_insulin;diabetespedigreefunction_per_age'), extraEntry = Entry('outcome', '10'))),
                                 #('anova', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '8', entry = Entry('pregnancies;glucose;bloodpressure;skinthickness;insulin;bmi;age;diabetespedigreefunction;outcome', 'pregnancies_per_glucose;pregnancies_per_bloodpressure;pregnancies_per_skinthickness;pregnancies_per_insulin;pregnancies_per_bmi;pregnancies_per_age;pregnancies_per_diabetespedigreefunction;glucose_per_pregnancies;glucose_per_bloodpressure;glucose_per_skinthickness;glucose_per_insulin;glucose_per_bmi;glucose_per_age;glucose_per_diabetespedigreefunction;bloodpressure_per_pregnancies;bloodpressure_per_glucose;bloodpressure_per_skinthickness;bloodpressure_per_insulin;bloodpressure_per_bmi;bloodpressure_per_age;bloodpressure_per_diabetespedigreefunction;skinthickness_per_bloodpressure;skinthickness_per_pregnancies;skinthickness_per_glucose;skinthickness_per_insulin;skinthickness_per_bmi;skinthickness_per_age;skinthickness_per_diabetespedigreefunction;insulin_per_bloodpressure;insulin_per_pregnancies;insulin_per_glucose;insulin_per_skinthickness;insulin_per_bmi;insulin_per_age;insulin_per_diabetespedigreefunction;bmi_per_bloodpressure;bmi_per_pregnancies;bmi_per_glucose;bmi_per_skinthickness;bmi_per_insulin;bmi_per_age;bmi_per_diabetespedigreefunction;age_per_bloodpressure;age_per_pregnancies;age_per_glucose;age_per_skinthickness;age_per_insulin;age_per_bmi;age_per_diabetespedigreefunction;diabetespedigreefunction_per_bloodpressure;diabetespedigreefunction_per_pregnancies;diabetespedigreefunction_per_glucose;diabetespedigreefunction_per_skinthickness;diabetespedigreefunction_per_insulin;diabetespedigreefunction_per_bmi;diabetespedigreefunction'), extraEntry = Entry('outcome', '10'))),
                                 ('anova', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '8', entry = Entry('pregnancies;glucose;bloodpressure;skinthickness;insulin;bmi;age;diabetespedigreefunction;outcome', 'glucose_per_skinthickness;glucose_per_insulin;glucose_per_diabetespedigreefunction;glucose_per_age;glucose_per_pregnancies;glucose_per_bmi;glucose_per_bloodpressure;skinthickness_per_glucose;skinthickness_per_insulin;skinthickness_per_diabetespedigreefunction;skinthickness_per_age;skinthickness_per_pregnancies;skinthickness_per_bmi;skinthickness_per_bloodpressure;insulin_per_glucose;insulin_per_skinthickness;insulin_per_diabetespedigreefunction;insulin_per_age;insulin_per_pregnancies;insulin_per_bmi;insulin_per_bloodpressure;diabetespedigreefunction_per_glucose;diabetespedigreefunction_per_skinthickness;diabetespedigreefunction_per_insulin;diabetespedigreefunction_per_age;diabetespedigreefunction_per_pregnancies;diabetespedigreefunction_per_bmi;diabetespedigreefunction_per_bloodpressure;age_per_glucose;age_per_skinthickness;age_per_insulin;age_per_diabetespedigreefunction;age_per_pregnancies;age_per_bmi;age_per_bloodpressure;pregnancies_per_glucose;pregnancies_per_skinthickness;pregnancies_per_insulin;pregnancies_per_diabetespedigreefunction;pregnancies_per_age;pregnancies_per_bmi;pregnancies_per_bloodpressure;bmi_per_glucose;bmi_per_skinthickness;bmi_per_insulin;bmi_per_diabetespedigreefunction;bmi_per_age;bmi_per_pregnancies;bmi_per_bloodpressure;bloodpressure_per_glucose;bloodpressure_per_skinthickness;bloodpressure_per_insulin;bloodpressure_per_diabetespedigreefunction;bloodpressure_per_age;bloodpressure_per_pregnancies;bloodpressure_per_bmi'), extraEntry = Entry('outcome', '10'))),
                                 ('show_corr_second', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '7', entry = Entry('outcome', None), extraEntry = Entry(None, None))),
                                 #('profiling', GraphOperation(self.dataset, isPrompt = False, option = '11', entry = Entry('output.html', None))),
                                 #('show_values_left', DataOperation(self.dataset, isPrompt = False, firstOption = '2', secondOption = '2', entry = Entry('left', None), extraEntry = Entry(None, None))),
                                 #('logistic_regression_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '1', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('svm_linear_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '2', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('svm_non_linear_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '3', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('decision_tree_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '4', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('knn_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '5', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('naive_bayes_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '6', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('random_forest_CV', SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = '7', entry = Entry('left', 'average_monthly_hours_per_satisfaction_level;time_spend_company;average_monthly_hours;promotion_last_5years;salary;work_accident;average_monthly_hours_per_time_spend_company;satisfaction_level;satisfaction_level_per_time_spend_company'), extraEntry = Entry(1, None)))
                                 #('logistic_regression', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '1', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('svm_linear', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '2', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('svm_non_linear', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '3', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('decision_tree', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '4', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('knn', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '5', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('naive_bayes', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '6', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('random_forest', SupervisedLearningOperation(self.dataset, isPrompt = False, option = '7', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('kmeans', UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = '1', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('spectral_clustering', UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = '2', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('mean_shift', UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = '3', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('dbscan', UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = '4', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('birch', UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = '5', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('preprocessing_logistic_regression', UnsupervisedLearningPreProcessingOperation(self.dataset, isPrompt = False, option = '1', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('preprocessing_naive_bayes', UnsupervisedLearningPreProcessingOperation(self.dataset, isPrompt = False, option = '2', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #best
                                 ('preprocessing_decision_tree', UnsupervisedLearningPreProcessingOperation(self.dataset, isPrompt = False, option = '3', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;bloodpressure_per_age')))
                                 #('preprocessing_decision_tree', UnsupervisedLearningPreProcessingOperation(self.dataset, isPrompt = False, option = '3', entry = Entry('outcome', 'glucose;bmi;age;pregnancies;skinthickness;insulin;bloodpressure;diabetespedigreefunction;skinthickness_per_diabetespedigreefunction;skinthickness_per_glucose')))
                                ])
        
        num_pipeline.fit_transform(self.dataset.dataframe)
            
class Main:
    def __init__(self):
        self.dataset = Dataset(isPrompt = False, option = '1', entry = Entry(None, None))
        self.metadata = MetadataOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None))
        self.data = DataOperation(self.dataset, isPrompt = False, firstOption = None, secondOption = None, entry = Entry(None, None), extraEntry = Entry(None, None))
        self.graph = GraphOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None))
        self.supervisedCV = SupervisedLearningCrossValidationOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None), extraEntry = Entry(None, None))
        self.supervised = SupervisedLearningOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None))
        self.unsupervised = UnsupervisedLearningOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None))
        self.unsupervisedPreProcessing = UnsupervisedLearningPreProcessingOperation(self.dataset, isPrompt = False, option = None, entry = Entry(None, None))
        self.pipeline = PipelineOperation(self.dataset)

    def printMenu(self):
        print('1 - Dataset operations')
        print('2 - Metadata operations')
        print('3 - Data operations')
        print('4 - Graph operations')
        print('5 - Supervised Learning Cross Validation operations')
        print('6 - Supervised Learning operations')
        print('7 - Unsupervised Learning operations')
        print('8 - Unsupervised Learning PreProcessing operations')
        print('9 - Execute pre-defined pipeline operations')
        print('10 - Clear output')
        print('11 - Exit')

    def executeFlow(self):
        while True:
            try:
                self.printMenu()
                option = input('Choose your option:')

                if(option == '1'):
                    self.dataset.isPrompt = True
                    self.dataset.transform(None)
                elif(option == '2'):
                    self.metadata.isPrompt = True
                    self.metadata.transform(self.dataset.dataframe)
                elif(option == '3'):
                    self.data.isPrompt = True
                    self.data.transform(self.dataset.dataframe)
                elif(option == '4'):
                    self.graph.isPrompt = True
                    self.graph.transform(self.dataset.dataframe)
                elif(option == '5'):
                    self.supervisedCV.isPrompt = True
                    self.supervisedCV.transform(self.dataset.dataframe)
                elif(option == '6'):
                    self.supervised.isPrompt = True
                    self.supervised.transform(self.dataset.dataframe)
                elif(option == '7'):
                    self.unsupervised.isPrompt = True
                    self.unsupervised.transform(self.dataset.dataframe)
                elif(option == '8'):
                    self.unsupervisedPreProcessing.isPrompt = True
                    self.unsupervisedPreProcessing.transform(self.dataset.dataframe)
                elif(option == '9'):
                    self.dataset.isPrompt = False
                    self.dataset.option = '1'
                    self.pipeline.transform()
                elif(option == '10'):
                    clear_output()
                elif(option == '11'):
                    break
            except:
                print('An error occurred', sys.exc_info())

#/Users/hugosilva/Documents/Mestrado/2122/Seminarios Industriais/Artigo/Dataset/diabetes.csv
Main().executeFlow()

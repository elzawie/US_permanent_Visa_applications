# Basic modules for dataframe manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import iqr, norm, skew
from scipy.special import boxcox1p
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import math


def lower_cols(dataframe):
	"""
	Function used to convert column headings to lower case
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""
	dataframe.columns = [x.lower() for x in dataframe.columns]


def obj_to_cat(dataframe):
   """
	Function used to convert objects(strings) into categories
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""

   for n, c in dataframe.items():
        if is_string_dtype(c):
            dataframe[n] = c.astype('category').cat.as_ordered()
	


def fill_missing_nums(dataframe):    
    """
	 Function used to impute missing numerical values with column's median
	
	 Parameters:
	
	 dataframe - just as the parameter name implies, expects dataframe object
	
   	 """
    
    for n, c in dataframe.items(): 
        if is_numeric_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.median())



def fill_missing_cats(dataframe):
    """
    Function used to impute missing categorical values with column's mode
	
    Parameters:
	
    dataframe - just as the parameter name implies, expects dataframe object
	
    """
    for n, c in dataframe.items():
        if is_categorical_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.mode()[0])
	


def display_cols(dataframe, type = 'category', num_samples = 7):
	"""
	Function used to display columns of desired data type
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	type - data type we are looking for
	num_samples - number of rows to display
	
	"""
	mask = dataframe.dtypes == type
	return dataframe.loc[:, mask].sample(num_samples)
	


def display_nums_stats(dataframe):
    """
    Function used to calculate basic statistics of numerical columns.
	
    Parameters:
	
    dataframe - just as the parameter name implies, expects dataframe object
	
    """

    numericals = []
    for n, c in dataframe.items():
	    if is_numeric_dtype(c):
		    numericals.append(n)
    return dataframe[numericals].describe()


	
def outliers_by_col(dataframe, train_last_idx , multiplier = 1.5, plot_results = True, outliers_dictionary = False):
	"""
	Function used to determine outliers in each column.
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	multiplier - value used for calculating Tukey's Interquartile Range. By default we assume that all values lower than Q1 - (1.5 * IQR)  or greather than Q3 + (1.5 * IQR) are outliers
	plot_results - by default set to True. As a result boxplots for all columns with outliers will be plotted
	outliers_dictionary - by default set to False. If True, dictionary with column names as keys and lists of row indexes containing outliers as values will be returned 
	
	"""
	
	outliers_dict = {}
	for column in dataframe.columns:
			if is_numeric_dtype(dataframe[column][:train_last_idx]):
				iq_range = iqr(dataframe[column][:train_last_idx])
				q1 = np.percentile(dataframe[column][:train_last_idx], 25)
				q3 = np.percentile(dataframe[column][:train_last_idx], 75)
				lower_bound = q1 - (multiplier * iq_range)
				upper_bound = q3 + (multiplier * iq_range)
				select_indices = list(np.where((dataframe[column][:train_last_idx] < lower_bound) | (dataframe[column][:train_last_idx] > upper_bound))[0])
				if len(select_indices) > 0 :
					outliers_dict[column] = select_indices

	
	if plot_results == True:
		plot_categoricals(dataframe[:train_last_idx], outliers_dict.keys(), kind = 'box', figsize = (20,10))
		
	if outliers_dictionary == True:
		return outliers_dict



	
def nominalnums_to_cat(dataframe, unique_values_split = 30,  boundary = 10):
	"""
	Function for converting nominal numerical features into categorical variables. 
	
	Parameters:
	
	dataframe -just as the parameter name implies, expects dataframe object
	unique_values_split - number of unique values to treat a variable as a categorical one. By default, variable's data type will be changed to 'category' if it has less than 30 unique values
	boundary - decision boundary determining which variable names will be returned in list for further check. By default, all variables which take more than 10 unique values will be returned
	
	"""
	cols_to_verify = []
	for col in dataframe.columns:
		if is_numeric_dtype(dataframe[col]):
			length = len(dataframe[col].value_counts())
			if ((length < unique_values_split) and ('area' not in col)):
				dataframe[col] = dataframe[col].astype('category')
				if (length > boundary):
					cols_to_verify.append(col)
	return(cols_to_verify)
	

def display_plots(dataframe, columns, kind = 'count', figsize = (20,10)):
	"""
	Function for plotting suspicious categorical columns.

	Parameters:

	dataframe - just as the parameter name implies, expects dataframe object
	columns - list of columns or dictionary keys, e.g. list of columns returned by 'nominalnums_to_cat' function
	kind - by default set to 'count' to display countplots for given columns. If 'box' will be used as a value then function will display box plots. 

	"""
	
	length = len(columns)
	if length <= 6:
		plt.figure(figsize=figsize)
	elif length > 6 and length <= 12:
		plt.figure(figsize = next((x, int(y*2)) for x,y in [figsize]))
	elif length > 12 and length <= 18:
		plt.figure(figsize = next((x, int(y*3)) for x,y in [figsize]))
	elif length > 18 and length <= 24:
		plt.figure(figsize = next((x, int(y*4)) for x,y in [figsize]))
	elif length > 24 and length <= 30:
		plt.figure(figsize = next((x, int(y*5)) for x,y in [figsize]))	
	for ix, col in enumerate(columns):
		plt.subplot(np.ceil(length/3), 3, ix+1)
		if kind == 'count':
			sns.countplot(dataframe[col])
		elif kind == 'box':
			sns.boxplot(dataframe[col])
		elif kind == 'dist':
			sns.distplot(dataframe[col], fit = norm)
			
			
def binarize_numericals(dataframe, columns):
	"""
	Function for creating binomial categorical variables from unequally distributed numerical variables, all values equal to 0 will be denoted as 0 and those greater than 0 will be marked as 1. After conversion, all input variables will be dropped from dataframe.

	Parameters:

	dataframe - just as the parameter name implies, expects dataframe object
	columns - list of columns or dictionary keys to convert

	"""
	for col in columns:
		dataframe[col+'_bin'] = np.where(dataframe[col] > 0, 1, 0)
		dataframe[col+'_bin'] = dataframe[col+'_bin'].astype('category')

	dataframe.drop(labels= columns, axis=1, inplace = True)

		
		
def get_codes(dataframe):
	"""
	Function for converting values of categorical variables into numbers.
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""
	for column in dataframe.columns:
		if is_categorical_dtype(dataframe[column]):
			dataframe[column] = dataframe[column].cat.codes
			

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse(x,y): return math.sqrt(((x-y)**2).mean())


def rmsle_cv(model, trainingset, target, n_folds):
	
    rmse= np.sqrt(-cross_val_score(model, trainingset, target, scoring="neg_mean_squared_error", cv = n_folds, n_jobs = -1))
    return(rmse)
	

def print_score(model, trainingset, target, scoring_func = 'rmse', n_folds = None):
	"""
	Function used for checking the accuracy of the regression model.
	
	Parameters:
	
	model -just as the parameter name implies, expects model object
	trainingset - training dataset
	target - target variable
	scoring_func - scoring function to assess the model's performance. By default RMSE will be used
	
	"""
	if scoring_func == 'rmse':
		X_train, X_val, y_train, y_val = train_test_split(trainingset, target, test_size = 0.2, random_state = 123, shuffle = True)
		res = [rmse(model.predict(X_train), y_train), rmse(model.predict(X_val), y_val), model.score(X_train, y_train), model.score(X_val, y_val)]
		print('Training RMSE: {0:.3f} | Testing RMSE: {1:.3f} | Training R^2: {2:.3f} | Testing R^2: {3:.3f}'.format(res[0], res[1], res[2], res[3]))
	
	elif scoring_func == 'rmsle':
		X_train, X_val, y_train, y_val = train_test_split(trainingset, target, test_size = 0.2, random_state = 123, shuffle = True)
		res = [rmsle(model.predict(X_train), y_train), rmsle(model.predict(X_val), y_val), model.score(X_train, y_train), model.score(X_val, y_val)]
		print('Training RMSLE: {0:.3f} | Testing RMSLE: {1:.3f} | Training R^2: {2:.3f} | Testing R^2: {3:.3f}'.format(res[0], res[1], res[2], res[3]))
		
	elif scoring_func == 'rmsle_cv':	
		if isinstance(trainingset, np.ndarray) & isinstance(target, np.ndarray):
			n_folds = n_folds
			kf = KFold(n_folds, shuffle=True, random_state=123).get_n_splits(trainingset)
			res = rmsle_cv(model, trainingset, target, n_folds )
			model = model.fit(trainingset, target)
			print('Average cross-validated RMSE: {0:.4f}  |  Standard Deviation of RMSE: {1:.4f}  |  Training R^2: {2:.3f}'.format(res.mean(), res.std(), model.score(trainingset, target)))
			
		else:
			n_folds = n_folds
			kf = KFold(n_folds, shuffle=True, random_state=123).get_n_splits(trainingset.values)
			res = rmsle_cv(model, trainingset, target, n_folds )
			model = model.fit(trainingset, target)
			print('Average cross-validated RMSE: {0:.4f}  |  Standard Deviation of RMSE: {1:.4f}  |  Training R^2: {2:.3f}'.format(res.mean(), res.std(), model.score(trainingset, target)))
			
		
		
	
def plot_feat_imp(model, dataframe, boundary = 15, best_features = False):

	"""
	Function used for plotting the most important features found by model.
	
	Parameters:
	
	model - just as the parameter name implies, expects model object
	dataframe - just as the parameter name implies, expects dataframe object
	boundary - number of features we would like to plot
	
	"""
	indices = np.argsort(model.feature_importances_)[::-1][:boundary]
	best_features_list = [col for col in dataframe.columns[indices]]

	fig = plt.figure(figsize=(9, 12))
	p = sns.barplot(y=dataframe.columns[indices][:boundary], x = model.feature_importances_[indices][:boundary], orient='h')
	p.set_xlabel("Relative importance",fontsize=12)
	p.set_ylabel("Features",fontsize=12)
	p.tick_params(labelsize=10)
	p.set_title("Feature importances")
	for i, v in enumerate(model.feature_importances_[indices][:boundary]):
		plt.text(v, i, ""+str(np.round(v,3)), color='#e59471', va='center', fontweight='bold')

	plt.show()
	
	if best_features == True:

		return best_features_list
	
	
def drop_best_feats(model, features, X, y, scoring_func):
	"""
	Function used to evaluate the performance of a model without best features. In each iteration model is dropping one of the best features.
	
	Parameters:
	
	model - just as the parameter name implies, expects model object
	features - list of features to drop
	X - training features vector
	y - training target vector
	scoring_func - function to be used for evaluation:  'rmse' or 'rmsle' 
	
	"""
	for feature in features:
		X_sub = X.drop(feature, axis = 1)
		X_train, X_val, y_train, y_val = train_test_split(X_sub, y, test_size = 0.2, random_state = 123, shuffle = False)
		model.fit(X_train, y_train)
		print('Dropped feature: {} '.format(feature))
		print_score(model, X_train, X_val, y_train, y_val, scoring_func = scoring_func)
		print('\n')

		

def plot_distqq(x, dataframe):
	"""
	Function used to plot distribution of the desired numerical variable with normal distribution overlayed and quantile-quantile plot.
	
	Parameters:
	
	x - numerical variable to plot
	dataframe - just as the name implies, expects dataframe object

	
	"""

	sns.distplot(dataframe[x], fit = norm)
	# Get the fitted parameters used by the function
	mu, sigma = norm.fit(dataframe[x])
	
	# Now plot the distribution
	plt.legend(['Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
	plt.ylabel('Frequency')
	plt.title('{} distribution'.format(x))
	
	#Get also the QQ-plot
	fig = plt.figure()
	res = stats.probplot(dataframe[x], plot=plt)
	plt.show()
	

def percent_missing(dataframe, ascending = False, quantity = 30):
	"""
	Function used to calculate the percentage of missing data. As a result returns dataframe.
	
	Parameters:
	
	dataframe - expects dataframe object
	ascending - whether values should be sorted in ascending or descending order. By default dataframe will be sorted descendingly.
	quantity - number of variables to display

	
	"""
	missing = (dataframe.isnull().sum() / len(dataframe)) * 100
	missing = missing.drop(missing[missing == 0].index).sort_values(ascending = ascending) [:quantity]
	missing = pd.DataFrame({'Missing Ratio' : missing})
	return missing

	
def plot_bar(x, y, xlabel, ylabel, title, figsize = (12, 8)):
	"""
	Function used to display a barplot of desired variables.
	
	Parameters:
	
	x - feature variable
	y - values of desired feature variable
	figsize - plot size. By default width = 12 and height = 8
	xlabel - just as the name implies, expects label for x-axis
	ylabel - just as the name implies, expects label for y-axis
	title - just as the name implies, expects plot title

	
	"""

	f, ax = plt.subplots(figsize = figsize)
	plt.xticks(rotation = '90')
	sns.barplot(x = x, y = y)
	plt.xlabel(xlabel, fontsize = 13)
	plt.ylabel(ylabel, fontsize = 13)
	plt.title(title, fontsize = 13)
	
def calculate_skewness(dataframe):
	"""
	Function used to calculate skewness across numerical features. As a result returns dataframe.
	
	Parameters:
	
	dataframe - just as the name implies, expects dataframe object
	
	"""

	numeric_feats = dataframe.dtypes[(dataframe.dtypes != "category") & (dataframe.dtypes != "object")].index
	# Check the skew of all numerical features
	skewed_feats = dataframe[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
	skewness = pd.DataFrame({'Skewness' :skewed_feats})
	return skewness
	

def box_cox_transform(dataframe, skewnesses, lamb):
	"""
	Function used to apply box-cox transformation for highly skewed features to make them look more normally distributed.
	
	Parameters:
	
	dataframe - just as the name implies, expects dataframe object
	skewenesses - expects dataframe object with calculated skewnesses. Use output from 'calculate_skeweness' function.
	lamb - lambda value to be used with box-cox transformation. Be defalt boxcox1p is used as it is better for smaller x values. Setting lamb = 0 is equivalent to log1p.
	
	"""
	
	skewness = skewnesses[abs(skewnesses) > 0.75]
	skewed_features = skewness.index
	lamb = lamb 
	for feature in skewed_features:
		dataframe[feature] = boxcox1p(dataframe[feature], lamb)
		

class AveragedScorer(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
		
		
class StackedAveragedScorer(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=123)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
		
		
def reg_importances(model, dataframe, boundary = None, plot = False, figsize = (9, 24), step_name = None):
	
	
	"""
	Function used for plotting the most important features found by regression model.
	
	Parameters:
	
	model - trained model 
	dataframe - just as the parameter name implies, expects dataframe object
	boundary - number of features we would like to return/plot. By default all features with absolute value > 0 will be returned
	plot - whether to plot feature importances or not. By default "False"
	figsize - just as the parameter name implies, enables customizing plot size. By default width = 9 and height = 24
	step_name - argument required if model was used in Pipeline object hence is of Pipeline type and expects exact step name to retrieve coefficients
	
	"""

	
	
	if isinstance(model, Pipeline):
		coefs_pos = [coef for coef in model.named_steps[step_name].coef_ if coef > 0]
		coefs_neg = [coef for coef in model.named_steps[step_name].coef_ if coef < 0]
		
		if boundary == None:
			pos_imps = pd.DataFrame({'Importance': coefs_pos }, index = dataframe.columns[model.named_steps[step_name].coef_ > 0]).sort_values('Importance', ascending = False)
			neg_imps = pd.DataFrame({'Importance': coefs_neg }, index = dataframe.columns[model.named_steps[step_name].coef_ < 0]).sort_values('Importance', ascending = True)
		else:
			pos_imps = pd.DataFrame({'Importance': coefs_pos }, index = dataframe.columns[model.named_steps[step_name].coef_ > 0]).sort_values('Importance', ascending = False)[:boundary]
			neg_imps = pd.DataFrame({'Importance': coefs_neg }, index = dataframe.columns[model.named_steps[step_name].coef_ < 0]).sort_values('Importance', ascending = True)[:boundary]
			
	else:
		coefs_pos = [coef for coef in model.coef_ if coef > 0]
		coefs_neg = [coef for coef in model.coef_ if coef < 0]
		
		if boundary == None:
			pos_imps = pd.DataFrame({'Importance': coefs_pos }, index = dataframe.columns[model.coef_ > 0]).sort_values('Importance', ascending = False)
			neg_imps = pd.DataFrame({'Importance': coefs_neg }, index = dataframe.columns[model.coef_ < 0]).sort_values('Importance', ascending = True)
		else:
			pos_imps = pd.DataFrame({'Importance': coefs_pos }, index = dataframe.columns[model.coef_ > 0]).sort_values('Importance', ascending = False)[:boundary]
			neg_imps = pd.DataFrame({'Importance': coefs_neg }, index = dataframe.columns[model.coef_ < 0]).sort_values('Importance', ascending = True)[:boundary]
	
	if plot == True:
		fig = plt.figure(figsize = figsize)
		plt.subplot(211)	
		p = sns.barplot(y= pos_imps.index, x = pos_imps['Importance'], orient='h')
		p.set_xlabel("Relative importance",fontsize=12)
		p.set_ylabel("Features",fontsize=12)
		plt.title("Positive importances")
		for i, v in enumerate(pos_imps['Importance']):
			plt.text(v, i, ""+str(np.round(v,3)), color='#e59471', va='center', fontweight='bold')
			
		
		plt.subplot(212)
		p = sns.barplot(y=neg_imps.index, x = neg_imps['Importance'], orient='h')
		p.set_xlabel("Relative importance",fontsize=12)
		p.set_ylabel("Features",fontsize=12)
		plt.title("Negative importances")

		for i, v in enumerate(neg_imps['Importance']):
			plt.text(v, i, ""+str(np.round(v,3)), color='#e59471', va='center', ha = 'right', fontweight='bold')
			
		plt.subplots_adjust(top = 0.9)
		plt.show()
		
	return pos_imps, neg_imps
		


def plot_counts(dataframe, x, hue = None, boundary = None, ascending = False, figsize = (12,8), xlabel = 'Feature', ylabel = 'Counts', title = '', fontsize = 13, rotation = None, palette = 'Blues', annotations = None, style = 'default'):
	"""
	Function used to display a countplot of desired variables.
	
	Parameters:
	
	dataframe = just as the name implies, expects datatframe object
	x - feature variable
	hue - categorical variable in data to map plot aspects to different colors
	boundary - number of unique feature values to display. By default set to None. Useful when there are dozens of possible values
	ascending - used when boundary is specified. By default set to None as order depends on order of the hue variable. 
	figsize - plot size. By default width = 12 and height = 8
	xlabel - just as the name implies, expects label for x-axis
	ylabel - just as the name implies, expects label for y-axis
	title -  just as the name implies, expects plot title

	"""
	f, ax = plt.subplots(figsize = figsize)
	
	if style != 'default':
		plt.style.use(style)

	if (boundary != None) & (ascending == False): 
		sns.countplot(x = x, hue = hue, data = dataframe, palette = palette, order = dataframe[x].value_counts().iloc[:boundary].index)
	elif (boundary != None) & (ascending == True):
		sns.countplot(x = x, hue = hue, data = dataframe, palette = palette, order = dataframe[x].value_counts(ascending = True).iloc[:boundary].index)
	else:
		sns.countplot(x = x, hue = hue, data = dataframe, palette = palette, order = dataframe[x].value_counts().index)
	if rotation != None:
		plt.xticks(rotation = rotation)
	plt.xlabel(xlabel, fontsize = fontsize)
	plt.ylabel(ylabel, fontsize = fontsize)
	plt.title(title, fontsize = 1.3 * fontsize)
	plt.xticks(fontsize= fontsize*0.8)
	plt.yticks(fontsize=fontsize*0.8)
	plt.show()
	

def plot_pie(dataframe, x, figsize = (12,12), boundary = 5, palette = sns.color_palette("husl", 8), title = '', pctdistance = 0.9, fontsize = 13,  labeldistance = 1.05, startangle = 140 ):
	"""
	Function used to display a pieplot of desired variable.
	
	Parameters:
    
	dataframe = just as the name implies, expects datatframe object
	x - feature variable
	boundary - number of unique feature values to display. By default set to 5. Useful when there are dozens of possible values.
	figsize - plot size. By default width = 12 and height = 12
	palette - color palette to use
	pctdistance - the ratio between the center of each pie slice and the start of the text generated by autopct
	labeldistance - the radial distance at which the pie labels are drawn
	starangle - rotates the start of the pie chart by angle degrees counterclockwise from the x-axis


	"""
    
	f, ax = plt.subplots(figsize = figsize)
	plt.pie(dataframe[x].value_counts()[:boundary], labels = dataframe[x].value_counts()[:boundary].index, pctdistance = pctdistance, autopct='%1.1f%%', colors = palette, labeldistance = labeldistance, startangle = startangle)
	plt.title(title, fontsize = 1.3 * fontsize)
	plt.show()
	



def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)






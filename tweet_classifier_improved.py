from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import string
import operator
import pathlib
import nltkclean_improved as cleanTweets

class tweet_classifier():

	def __init__(self, tweets, params):

		if len(params) != 6 or params[0:4] == [True, True, True, True, True]:
			raise ValueError('Incorrect parameters.')

		self.df = tweets
		self.params = params
		self.politicians = set(pd.read_csv('politicians.csv', encoding='cp1252')['name'])
		self.issues = set(pd.read_csv('issues.csv')['topic'])
		self.allfeatures = pd.read_csv('allwords_standardcleaning.csv', encoding='cp1252')['Feature'].tolist()
		self.cleaned = False
		self.featureSelection = ''.join(str(x)[0] if type(x) == bool else str(x) for x in params)
		self.modelReport = None
		self.predictedLabelNB = []
		self.predictedLabelSVM = []
		self.correctLabelNB = []
		self.correctLabelSVM = []
		self.accuraciesNB = []
		self.accuraciesSVM = []

	'''Cleans tweets with given parameters in self.params. params is as follows: 
	[removeText?, removeHashtags?, remove@s? removePoliticans?, removeIssues?, number of features to remove] 
	(e.g. removeHashtags=True will remove hashtags)'''
	def clean(self, generateCSV = False):

		self.df = cleanTweets.clean(self.df, self.politicians, self.issues, self.allfeatures, removetext = self.params[0], removehashtags = self.params[1], removeats = self.params[2], removepoliticians = self.params[3], removeissues = self.params[4], numfeatures = self.params[5])
		self.cleaned = True

		if generateCSV:
			cleandFileName = 'cleanedTweets{}{}{}{}{}{}.csv'.format(str(self.params[0])[0], str(self.params[1])[0], str(self.params[2])[0], str(self.params[3])[0], str(self.params[4])[0], str(self.params[5]))
			pathlib.Path('./{}'.format(self.featureSelection)).mkdir(exist_ok=True) 
			self.df.to_csv('{}/{}'.format(self.featureSelection, cleandFileName), encoding='utf-8')
			print('CSV outputted to {} folder'.format(self.featureSelection))

	#Custom scorer for averaging precision, recall, f1, and accuracy across 5-folds.
	def classification_report_with_accuracy_score(self, y_true, y_pred):

		if self.modelReport == 'NB':
			self.predictedLabelNB.extend(y_pred)
			self.correctLabelNB.extend(y_true)
			self.accuraciesNB.append(accuracy_score(y_true, y_pred))
			return accuracy_score(y_true, y_pred)
		else:
			self.predictedLabelSVM.extend(y_pred)
			self.correctLabelSVM.extend(y_true)
			self.accuraciesSVM.append(accuracy_score(y_true, y_pred))
			return accuracy_score(y_true, y_pred)

	#Generates classification report (precision, recall, f1, accuracy) of Naive Bayes and SVM across 5-folds.
	def generate_report(self):

		if not self.cleaned:
			print('WARNING: Clean before generating report.')

		print('Learning...')

		#Classification Report for 80/20 split one test.
		'''#Creating training/testing data and labels.
		docs_train = self.df['text'].tolist()[:433]
		docs_train_label = self.df['Party'].tolist()[:433]
		docs_test = self.df['text'].tolist()[-108:]
		docs_test_label = self.df['Party'].tolist()[-108:]

		#Naive Bayes
		text_clf_NB = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
		text_clf_NB.fit(docs_train, docs_train_label)
		predicted = text_clf_NB.predict(docs_test)
		print(predicted)
		print('NaiveBayesClassifier: ')
		print(np.mean(predicted == docs_test_label))

		#Classification report for Naive Bayes(precision, recall, etc.)
		print(metrics.classification_report(docs_test_label, predicted, target_names=self.df['Party'].unique()))

		#SVM
		text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
		text_clf_SVM.fit(docs_train, docs_train_label)
		predicted = text_clf_SVM.predict(docs_test)
		print('SVM: ')
		print(np.mean(predicted == docs_test_label))

		#Classification report for SVM(precision, recall, etc.)
		print(metrics.classification_report(docs_test_label, predicted, target_names=self.df['Party'].unique()))'''

		print('Performing 5-fold CV on Naive Bayes...')
		#5-fold Cross Validation with Naive Bayes
		self.modelReport = 'NB'
		text_clf_NB = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
		nested_score = cross_val_score(text_clf_NB, X=self.df.text, y=self.df.Party, cv=5, scoring=make_scorer(self.classification_report_with_accuracy_score))
		print('Classification Report for Naive Bayes across 5-folds: ')
		print(classification_report(self.correctLabelNB, self.predictedLabelNB))
		print('Accuracy: ', np.mean(self.accuraciesNB))
		print('Done')

		print('---------------------------------------------')

		print('Performing 5-fold CV on SVM...')
		#5-fold Cross Validation with SVM
		self.modelReport = 'SVM'
		text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
		nested_score = cross_val_score(text_clf_SVM, X=self.df.text, y=self.df.Party, cv=5, scoring=make_scorer(self.classification_report_with_accuracy_score))
		print('Classification Report for SVM across 5-folds: ')
		print(classification_report(self.correctLabelSVM, self.predictedLabelSVM))
		print('Accuracy: ', np.mean(self.accuraciesSVM))
		print('Done')

	#Generates features and weights(coefficients) into CSV; If idf=True, IDF for each feature output into CSV.
	def generate_features(self, weights=True, idf=False):

		if not self.cleaned:
			print('WARNING: Clean before generating features.')

		docs_train = self.df['text'].tolist()
		docs_train_label = self.df['Party'].tolist()

		if idf:
			#IDF Extraction
			vectorizer = TfidfVectorizer(min_df=1)
			X = vectorizer.fit_transform(docs_train)
			idf = vectorizer.idf_
			results = dict(zip(vectorizer.get_feature_names(), idf))
			sorted_results = sorted(results.items(), key=operator.itemgetter(1))
			sorted_results.reverse()
			sorted_resultsdf = pd.DataFrame(sorted_results)
			sorted_resultsdf.columns = ['Feature', 'IDF']
			pathlib.Path('./{}'.format(self.featureSelection)).mkdir(exist_ok=True) 
			sorted_resultsdf.to_csv('{}/idf_words_{}{}{}{}{}{}.csv'.format(self.featureSelection, str(self.params[0])[0], str(self.params[1])[0], str(self.params[2])[0], str(self.params[3])[0], str(self.params[4])[0], str(self.params[5])), encoding='utf-8')
			print('IDF outputted to {} folder'.format(self.featureSelection))

		if weights:
			#Features and coefficients extraction
			text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
			text_clf_SVM.fit(docs_train, docs_train_label)
			feats = text_clf_SVM.steps[0][1].get_feature_names()
			coeff = text_clf_SVM.steps[1][1].coef_.tolist()[0]

			newresults = dict(zip(feats, coeff))
			allfeatures = sorted(newresults.items(), key=operator.itemgetter(1))
			dem = sorted(newresults.items(), key=operator.itemgetter(1))[:50]
			rep = sorted(newresults.items(), key=operator.itemgetter(1))[-50:]

			demwordsdf = pd.DataFrame(dem)
			demwordsdf.columns = ['Feature', 'Coefficient']
			pathlib.Path('./{}'.format(self.featureSelection)).mkdir(exist_ok=True) 
			demwordsdf.to_csv('{}/demwords_coeff_{}{}{}{}{}{}.csv'.format(self.featureSelection, str(self.params[0])[0], str(self.params[1])[0], str(self.params[2])[0], str(self.params[3])[0], str(self.params[4])[0], str(self.params[5])), encoding='utf-8')
			repwordsdf = pd.DataFrame(rep)
			repwordsdf.columns = ['Feature', 'Coefficient']
			repwordsdf.sort_values("Coefficient", ascending=False, inplace=True)
			repwordsdf.to_csv('{}/repwords_coeff_{}{}{}{}{}{}.csv'.format(self.featureSelection, str(self.params[0])[0], str(self.params[1])[0], str(self.params[2])[0], str(self.params[3])[0], str(self.params[4])[0], str(self.params[5])), encoding='utf-8')
			allfeaturesdf = pd.DataFrame(allfeatures)
			allfeaturesdf.columns = ['Feature', 'Coefficient']
			allfeaturesdf.to_csv('{}/allwords_coeff_{}{}{}{}{}{}.csv'.format(self.featureSelection, str(self.params[0])[0], str(self.params[1])[0], str(self.params[2])[0], str(self.params[3])[0], str(self.params[4])[0], str(self.params[5])), encoding='utf-8')
			print('Features outputted to {} folder'.format(self.featureSelection))
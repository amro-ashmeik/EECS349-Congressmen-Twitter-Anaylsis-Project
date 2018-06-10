from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import string
import operator
import csv

def isdistrict(token):

	chars = list(token)

	if chars[0].isalpha() and chars[1].isalpha() and chars[2].isdigit() and chars[3].isdigit():
		return True
	else:
		return False



df = pd.read_csv('500tweetsfinal.csv', encoding='cp1252')

#Words to be entirely removed.
stop_words = set(stopwords.words('english'))
stop_words.update(['.pdf', '–', '—', '-', '…', '=', '#', '@', ',', "'", '?', '‘', '!', '``', '--', "'m", "''", '“', '”', '...', "n't", "'re", '.', '(', ')', ';', ':', '"', "'s", "'ll", "'ve", 'http', 'https', "’", 'republicans', 'republican', 'repubs', 'reps', 'Republican', 'Republicans', 'Repubs', 'Reps', 'GOP', 'repub', 'Repub', 'Dems', 'dems', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat'])

#Substrings to be removed.
substrings = ['—', '…', '=', 'bit.ly', '.pdf', '.html', 'pic', 'http', 'https', 'twitter', '.com', '.co', '...', '/', '.org', '.gov', 'www', 'republicans', 'republican', 'repubs', 'Republican', 'Republicans', 'Repubs', 'GOP', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat', '+', '-', '_']

#Tokenizing and text and applying cleaning (removing stopwords, substrings)
print ('Creating tokens and cleaning...')
for i in df.index:
	print(str(int((i + 1) / (546) * 100)) + '% Complete', end='\r')
	for column in df.columns:
		if column != 'Name' and column != 'Party' and not pd.isnull(df.loc[i, column]):
			alltokens = word_tokenize(df.at[i, column])
			tokens = []
			#tag = False
			for token in alltokens:
				token = token.lower()
				'''if token == '@':
					tag = True
					continue
				elif tag:
					tokens.append(token)
					tag = False
					continue
				else:
					continue'''
				if token in stop_words or token in string.punctuation:
					continue
				if 'status' in token and len(token) > 15:
					continue
				#Comment next 2 lines and uncomment lines 40-48 if you want to remove just substrings and not entire word containing substring.
				if any(substring in token for substring in substrings):
					continue
				'''while True:
					done = 0
					for substring in substrings:
						if substring in token:
							done = 1
							token = token.replace(substring, '')
					if done == 0:
						break'''
				if len(token) == 4:
					if isdistrict(token):
						continue
				tokens.append(token)
			conca = " ".join(x for x in tokens) #Untokenizing
			df.at[i, column] = conca

#Combining all 500 tweets into one text column.
print ('Concatenating text columns into one...')

cols = [x for x in df.columns if x != 'Name' and x != 'Party']

def concat_text(row, cols):
    # The real work is done here
    return "".join([" ".join([str(x) for x in y if x]) for y in row[cols].values])

df = df.groupby(["Name", 'Party']).apply(concat_text, cols).reset_index()
df.columns = ['Name', 'Party', 'text']

df = df[['Name', 'Party', 'text']]
df.dropna(subset=['text'], inplace=True)
df.drop(df[df['Party']  == 'I'].index, inplace=True) #Removing Independents
#df = shuffle(df)
df.to_csv('testagain.csv', encoding='utf-8')
#505 congressman left(503 without Independents)

#df = pd.read_csv('superclean.csv', encoding='cp1252')

print('Learning...')

#Creating training/testing data and labels.
docs_train = df['text'].tolist()[:352]
docs_train_label = df['Party'].tolist()[:352]
docs_test = df['text'].tolist()[-151:]
docs_test_label = df['Party'].tolist()[-151:]

#Naive Bayes
text_clf_NB = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
text_clf_NB.fit(docs_train, docs_train_label)
predicted = text_clf_NB.predict(docs_test)
print('NaiveBayesClassifier: ')
print(np.mean(predicted == docs_test_label))

#Classification report for Naive Bayes(precision, recall, etc.)
print(metrics.classification_report(docs_test_label, predicted, target_names=df['Party'].unique()))

#SVM
text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf_SVM.fit(docs_train, docs_train_label)
predicted = text_clf_SVM.predict(docs_test)
print('SVM: ')
print(np.mean(predicted == docs_test_label))

#Classification report for SVM(precision, recall, etc.)
print(metrics.classification_report(docs_test_label, predicted, target_names=df['Party'].unique()))

#5-fold Cross Validation with Naive Bayes
text_clf_NB = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
scores = cross_val_score(text_clf_NB, df.text, df.Party, cv=5)
print('5-fold CV with Naive Bayes: ')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#5-fold Cross Validation with SVM
text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
scores = cross_val_score(text_clf_SVM, df.text, df.Party, cv=5)
print('5-fold CV with SVM: ')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#--------------------------FEATURE EXTRACTION----------------------------------------

docs_train = df['text'].tolist()
docs_train_label = df['Party'].tolist()

#IDF Extraction
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(docs_train)
idf = vectorizer.idf_
results = dict(zip(vectorizer.get_feature_names(), idf))
sorted_results = sorted(results.items(), key=operator.itemgetter(1))
#sorted_results.reverse()
#print(sorted_results)


#Features and coefficients extraction
text_clf_SVM = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf_SVM.fit(docs_train, docs_train_label)
feats = text_clf_SVM.steps[0][1].get_feature_names()
coeff = text_clf_SVM.steps[1][1].coef_.tolist()[0]

newresults = dict(zip(feats, coeff))
dem = sorted(newresults.items(), key=operator.itemgetter(1))[:2000]
rep = sorted(newresults.items(), key=operator.itemgetter(1))[-2000:]

demwordsdf = pd.DataFrame(dem)
demwordsdf.columns = ['Feature', 'Coefficient']
demwordsdf.to_csv('demwords.csv')
repwordsdf = pd.DataFrame(rep)
repwordsdf.columns = ['Feature', 'Coefficient']
repwordsdf.sort_values("Coefficient", ascending=False, inplace=True)
repwordsdf.to_csv('repwords.csv')
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

'''df = pd.read_csv('500tweetsfinal.csv', encoding='cp1252')

#Words to be entirely removed.
stop_words = set(stopwords.words('english'))
stop_words.update(['.pdf', '–', '—', '-', '…', '=', '#', '@', ',', "'", '?', '‘', '!', '``', '--', "'m", "''", '“', '”', '...', "n't", "'re", '.', '(', ')', ';', ':', '"', "'s", "'ll", "'ve", 'http', 'https', "’", 'republicans', 'republican', 'repubs', 'reps', 'Republican', 'Republicans', 'Repubs', 'Reps', 'GOP', 'repub', 'Repub', 'Dems', 'dems', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat'])

#Substrings to be removed.
substrings = ['—', '…', '=', 'bit.ly', '.pdf', '.html', 'pic.', 'http', 'https', 'twitter', '.com', '.co', '...', '/', '.org', '.gov', 'www', 'republicans', 'republican', 'repubs', 'Republican', 'Republicans', 'Repubs', 'GOP', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat', '+', '-', '_']

#Tokenizing and text and applying cleaning (removing stopwords, substrings)
print ('Creating tokens and cleaning...')
for i in df.index:
	print(str(int((i + 1) / (546) * 100)) + '% Complete', end='\r')
	for column in df.columns:
		if column != 'Name' and column != 'Party' and not pd.isnull(df.loc[i, column]):
			alltokens = word_tokenize(df.at[i, column])
			tokens = []
			for token in alltokens:
				token = token.lower()
				if token in stop_words or token in string.punctuation:
					continue
				if 'status' in token and len(token) > 15:
					continue
				#Comment next 2 lines and uncomment lines 40-48 if you want to remove just substrings and not entire word containing substring.
				if any(substring in token for substring in substrings):
					continue
				while True:
					done = 0
					for substring in substrings:
						if substring in token:
							done = 1
							token = token.replace(substring, '')
					if done == 0:
						break
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
df = shuffle(df)
df.to_csv('testagain.csv', encoding='utf-8')
print(df)
#505 congressman left(503 without Independents)'''

df = pd.read_csv('superclean.csv', encoding='cp1252')
print(df)

print('Learning...')

#Creating training/testing data and labels.
docs_train = df['text'].tolist()[:352]
docs_train_label = df['Party'].tolist()[:352]
docs_test = df['text'].tolist()[-151:]
docs_test_label = df['Party'].tolist()[-151:]

#Naive Bayes
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
text_clf.fit(docs_train, docs_train_label)
predicted = text_clf.predict(docs_test)
print('NaiveBayesClassifier: ')
print(np.mean(predicted == docs_test_label))

#Classification report for Naive Bayes(precision, recall, etc.)
print(metrics.classification_report(docs_test_label, predicted, target_names=df['Party'].unique()))

#SVM
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf.fit(docs_train, docs_train_label)  
predicted = text_clf.predict(docs_test)
print('SVM: ')
print(np.mean(predicted == docs_test_label))

#Classification report for SVM(precision, recall, etc.)
print(metrics.classification_report(docs_test_label, predicted, target_names=df['Party'].unique()))

#5-fold Cross Validation with Naive Bayes
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', MultinomialNB())])
scores = cross_val_score(text_clf, df.text, df.Party, cv=5)
print('5-fold CV with Naive Bayes: ')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#5-fold Cross Validation with SVM
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf = Pipeline([('tfidfv', TfidfVectorizer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
scores = cross_val_score(text_clf, df.text, df.Party, cv=5)
print('5-fold CV with SVM: ')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Feature Extraction (WIP)
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(docs_train)
idf = vectorizer.idf_
results = dict(zip(vectorizer.get_feature_names(), idf))

best = sorted(results)
print(results)

#--------OTHER RANDOM STUFF--------------
#Creates tuples of text and label.
'''print ('Tokenizing text column...')
for i in df.index:
	tokens = word_tokenize(df.at[i, 'text'])
	for word in tokens:
		if word in stop_words:
			tokens.remove(word)
	df.at[i, 'text'] = tokens

df['tuple'] = df[['text','Party']].apply(tuple, axis=1)

tuples = df['tuple'].tolist()'''

'''count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['text'].tolist()[:373])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, df['Party'].tolist()[:373])

docs_new = df['text'].tolist()[-155:]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, category))

for doc, category in zip(df['text'].tolist()[:373], df['Party'].tolist()[:373]):
	print(doc, '   =>   ', category)'''

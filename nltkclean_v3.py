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
import time
import string
import operator
import csv

# FORMAT = [text,#,@,issues,politicians]
# True includes said item, False removes the time

combo_list = [[True,True,True,True,True],[True,True,False,False,False]]

df = pd.read_csv('500tweetsfinal.csv', encoding='cp1252')

names_in = csv.reader(open('wiki-output_gov_mod.csv', 'r'))
issues_in = csv.reader(open('Issues.csv', 'r'))

#Words to be entirely removed.
stop_words = set(stopwords.words('english'))
stop_words.update(['.pdf', '–', '—', '-', '…', '=', ',', "'", '?', '‘', '!', '``', '--', "'m", "''", '“', '”', '...', "n't", "'re", '.', '(', ')', ';', ':', '"', "'s", "'ll", "'ve", 'http', 'https', "’", 'republicans', 'republican', 'repubs', 'reps', 'Republican', 'Republicans', 'Repubs', 'Reps', 'GOP', 'repub', 'Repub', 'Dems', 'dems', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat'])

#Substrings to be removed.
substrings = ['—', '…', '=', 'bit.ly', '.pdf', '.html', 'pic', 'http', 'https', 'twitter', '.com', '.co', '...', '/', '.org', '.gov', 'www', 'republicans', 'republican', 'repubs', 'Republican', 'Republicans', 'Repubs', 'GOP', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat', '+', '-', '_']

# Compile list of politicians
politicians = []
for tkn_t2c_p in names_in:
	politicians.append(tkn_t2c_p[0])

# Compile list of issues
issues = []
for tkn_t2c_i in issues_in:
	issues.append(tkn_t2c_i[0])
# GRAB ISSUES

# MEGA ITERATOR THROUGH EACH COMBO
for combo in combo_list:
	# Feature Flags

	txt_flg = combo[0]
	ht_flg = combo[1]
	at_flg = combo[2]
	iss_flg = combo[3]
	pol_flg = combo[4]

	# File Flags

	file_marker = ""

	for flag in combo:
		first_letter = str(flag)[0]
		file_marker += first_letter

	# Next Word Flag

	ht_next_word_flag = False
	at_next_word_flag = False

	print('\nBEGIN ANALYSIS FOR: ' + file_marker + '\n')

	#Tokenizing and text and applying cleaning (removing stopwords, substrings)
	print ('Preprocessing for ' + file_marker + '\n')
	for i in df.index:
		print(str(int((i + 1) / (546) * 100)) + '% Complete', end='\r')
		for column in df.columns:
			if column != 'Name' and column != 'Party' and not pd.isnull(df.loc[i, column]):
				alltokens = word_tokenize(df.at[i, column])
				tokens = []
				for token in alltokens:
					token = token.lower()
					
					# NEXT WORD REMOVAL ROUTINE
					if ht_next_word_flag:

						# Removes hashtags specific to district
						token_length = len(token)
						if token_length in range(3,4) and token.isdigit():
							#print('REMOVE DISTRICT #')
							continue

						if ht_flg:
							tokens.append(token)
							#print('ADD NEXT WORD ' + token)

						ht_next_word_flag = False
						continue

					if at_next_word_flag:

						if at_flg:
							tokens.append(token)
							#print('ADD NEXT WORD ' + token)

						at_next_word_flag = False
						continue

					# STANDARD REMOVAL ROUTINES

					# Handles weird case
					if 'status' in token and len(token) > 15:
						#print('REMOVE WEIRD SHIT ')
						continue

					# Removes word containing substring
					if any(substring in token for substring in substrings):
						#print('REMOVE SUBSTRING ' + token)
						continue

					# Handles stop words
					if token in stop_words or token in string.punctuation:
						#print('REMOVE GARBAGE ' + token)
						continue

					"""
					# Removes substrings from a word
					while True:
						done = 0
						for substring in substrings:
							if substring in token:
								done = 1
								token = token.replace(substring, '')
						if done == 0:
							break
					"""

					# END STANDARD REMOVAL ROUTINES

					# OPTIONAL INCLUSION/REMOVAL SUBROUTINES

					# Hashtag

					#print('at_flg')

					#if ht_flg:
					if token == "#":
						ht_next_word_flag = True
						#print('REMOVE #')
						continue

					#print('at_flg')

					# @
					#if at_flg:
					if token == "@":
						at_next_word_flag = True
						#print('REMOVE @')
						continue


					#print('iss_flg')

					# Issues
					if iss_flg:
						if token in issues:
							tokens.append(token)
							#print('ADD ISSUE ' + token)
							continue

					#print('pol_flg')

					# Politicians
					if pol_flg:
						if token in politicians:
							tokens.append(token)
							#print('ADD POLITICIAN ' + token)
							continue

					#print('txt_flg')

					# Text
					if txt_flg:
						tokens.append(token)
						#print('ADD TEXT ' + token)
						#print(token)

					#print('\nREMOVED TOKEN ' + token + '\n')

					#print('restart\n')
					#time.sleep(1)
					# END OF OPTIONAL ROUTINES

					# END OF ROUTINES

				conca = " ".join(x for x in tokens) #Untokenizing
				df.at[i, column] = conca

	#Combining all 500 tweets into one text column.
	print ('Concatenating text columns into one...\n')

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
	df.to_csv('cleaned_output_' + file_marker + '.csv', encoding='utf-8')
	#505 congressman left(503 without Independents)

	#df = pd.read_csv('superclean.csv', encoding='cp1252')

	print('Learning for ' + file_marker + '\n')

	#Creating training/testing data and labels.
	docs_train = df['text'].tolist()[:352]
	docs_train_label = df['Party'].tolist()[:352]
	docs_test = df['text'].tolist()[-151:]
	docs_test_label = df['Party'].tolist()[-151:]

	print('Test for ' + file_marker + '\n')

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

	# Mod file names
	repub_file_name = 'repub_' + file_marker + '.csv'
	dem_file_name = 'dem_' + file_marker + '.csv'

	demwordsdf = pd.DataFrame(dem)
	demwordsdf.columns = ['Feature', 'Coefficient']
	demwordsdf.to_csv(dem_file_name)
	repwordsdf = pd.DataFrame(rep)
	repwordsdf.columns = ['Feature', 'Coefficient']
	repwordsdf.sort_values("Coefficient", ascending=False, inplace=True)
	repwordsdf.to_csv(repub_file_name)

	print('\nEND ANALYSIS FOR: ' + file_marker + '\n')

print('Finished...')
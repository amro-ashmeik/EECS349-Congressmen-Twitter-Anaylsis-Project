from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import string

def isdistrict(token):

	chars = list(token)

	if len(token) == 4:
		if chars[0].isalpha() and chars[1].isalpha() and chars[2].isdigit() and chars[3].isdigit():
			return True
		else:
			return False
	if len(token) == 3:
		if chars[0].isalpha() and chars[1].isalpha() and chars[2].isdigit():
			return True
		else:
			return False

def clean(tweets, politicians, issues, allfeatures, removetext = False, removehashtags = False, removeats = False, removepoliticians = False, removeissues = False, numfeatures = 0):

	df = tweets
	if numfeatures != 0:
		features_to_remove = set(allfeatures[:numfeatures] + allfeatures[-numfeatures:])
	else:
		features_to_remove = None
		
	#Words to be entirely removed.
	stop_words = set(stopwords.words('english'))
	stop_words.update(['.pdf', '–', '—', '-', '…', '=', '#', '@', ',', "'", '?', '‘', '!', '``', '--', "'m", "''", '“', '”', '...', "n't", "'re", '.', '(', ')', ';', ':', '"', "'s", "'ll", "'ve", 'http', 'https', "’", 'republicans', 'republican', 'repubs', 'reps', 'Republican', 'Republicans', 'Repubs', 'Reps', 'GOP', 'repub', 'Repub', 'Dems', 'dems', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat'])

	#Substrings to be removed.
	substrings = set(['—', '…', '=', 'bit.ly', '.pdf', '.html', 'pic', 'http', 'https', 'twitter', '.com', '.co', '...', '/', '.org', '.gov', 'www', 'republicans', 'republican', 'repubs', 'Republican', 'Republicans', 'Repubs', 'GOP', 'Democratic', 'democratic', 'democrats', 'Democrats', 'democrat', 'Democrat', '+', '-', '_'])

	#Tokenizing and text and applying cleaning (removing stopwords, substrings)
	print ('Creating tokens and cleaning...')

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

	for i in df.index:
		print(str(int((i + 1) / (546) * 100)) + '% Complete', end='\r')
		
		alltokens = word_tokenize(df.at[i, 'text'])
		tokens = []
		hashtag_flag = False
		at_flag = False
		for token in alltokens:
			token = token.lower()
			#Comment following block of code and uncomment lines 149-157 if you want to remove just substrings and not entire word containing substring.
			if any(substring in token for substring in substrings):
				if hashtag_flag or at_flag:
					hashtag_flag = False
					at_flag = False
				continue

			#Removes N features as specified in parameters.
			if features_to_remove:
				if token in features_to_remove:
					if hashtag_flag or at_flag:
						hashtag_flag = False
						at_flag = False
					continue

			#If removing text...
			if removetext:
				#...but NOT removing hashtags.
				if not removehashtags:
					if token == '#':
						hashtag_flag = True
						continue
					elif hashtag_flag:
						if isdistrict(token):
							hashtag_flag = False
							continue
						tokens.append(token)
						hashtag_flag = False
						continue

				#...but NOT removing @'s.
				if not removeats:
					if token == '@':
						at_flag = True
						continue
					elif at_flag:
						if isdistrict(token):
							at_flag = False
							continue
						tokens.append(token)
						at_flag = False
						continue

				#...but NOT removing politcians.
				if not removepoliticians:
					if token in politicians:
						if isdistrict(token):
							continue
						tokens.append(token)
						continue

				#...but NOT removing issues.
				if not removeissues:
					if token in issues:
						if isdistrict(token):
							continue
						tokens.append(token)
						continue
				continue

			if removehashtags:
				if token == '#':
					hashtag_flag = True
					continue
				elif hashtag_flag:
					hashtag_flag = False
					continue

			if removeats:
				if token == '@':
					at_flag = True
					continue
				elif at_flag:
					at_flag = False
					continue

			if token in stop_words or token in string.punctuation:
				continue

			if 'status' in token and len(token) > 15:
				continue

			if (removepoliticians and token in politicians) or (removeissues and token in issues):
				continue

			#Uncomment below and comment first block of code in this for loop if you want to only remove substring and not entire word containing substring.
			'''while True:
				done = 0
				for substring in substrings:
					if substring in token:
						done = 1
						token = token.replace(substring, '')
				if done == 0:
					break'''

			if isdistrict(token):
				continue
			tokens.append(token)

		conca = " ".join(x for x in tokens) #Untokenizing
		df.at[i, 'text'] = conca

	print('Done cleaning.')
	return df
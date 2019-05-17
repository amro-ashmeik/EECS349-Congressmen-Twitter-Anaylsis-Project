import tweet_classifier_improved as tc
import pandas as pd


#Parameters Order: removetext?, removehastags?, removeats?, removepoliticans?, removeissues?, num_features_to_remove

'''df = pd.read_csv('superduperclean.csv', encoding='cp1252') #superduperclean is equivalent to FFFFF0
tweetClf = tc.tweet_classifier(df, [True, False, True, True, True, 0])
tweetClf.clean(generateCSV=True)
tweetClf.generate_features()
tweetClf.generate_report()'''

#Example for running multiple feature selections.
combos = [[False, False, False, False, False, 0], 
		[True, False, True, True, True, 0], 
		[True, True, False, True, True, 0], 
		[True, True, True, False, True, 0], 
		[True, True, True, True, False, 0],
		[False, True, False, False, False, 0]]

for combo in combos:
	df = pd.read_csv('500tweetsfinal.csv', encoding='cp1252')
	df = tc.tweet_classifier(df, combo)
	df.clean(generateCSV=True)
	df.generate_features()
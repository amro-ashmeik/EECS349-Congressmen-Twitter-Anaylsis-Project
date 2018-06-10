import csv
from nltk.tokenize import word_tokenize

mode = "a"

txt_in = 'gov-input.txt'
csv_out = csv.writer(open('wiki-output_gov_mod.csv', mode), lineterminator='\n')

text = filter(None, (line.rstrip() for line in open(txt_in)))

filter_set = set()

for line in text:
	if line != None:
		whole_name = word_tokenize(line)
		for tkn in whole_name:
			lower_tkn = tkn.lower()
			print(lower_tkn)
			filter_set.add(lower_tkn)

for val in filter_set:
	name = [val]
	csv_out.writerow(name)
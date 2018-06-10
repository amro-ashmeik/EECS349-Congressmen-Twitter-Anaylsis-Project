import csv
import time
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# Set mode to "w" for write or "a" to append
mode = "w"

input_list = ['house-input.txt','senate-input.txt']
wiki_out = csv.writer(open('wiki-output_gov_mod.csv', mode), lineterminator='\n')

print("Beginning Parsing\n")

for file in input_list:
	wiki_in = open(file)

	print("Start of parsing " + file + "\n")
	print("Listing individual names being parsed...\n")

	time.sleep(1)

	wiki_soup = BeautifulSoup(wiki_in, 'html.parser')

	span_list = wiki_soup.find_all("span", "fn")

	a_lst = []

	for tag in span_list:
		a_lst.append(tag.find_all("a"))

	lst_names = []

	for tag in a_lst:
		name = tag[0].string
		lst_names.append(name)
		print(name)

	print('\nIndividual names all parsed. Now adding to CSV.\n')

	for val in lst_names:
		print(val)

	print('\n'+str(len(lst_names))+'\n')

	print('Separating names into lower case tokens')

	filter_set = set()

	for val in lst_names:
		whole_name = word_tokenize(val)
		for tkn in whole_name:
			lower_tkn = tkn.lower()
			print(lower_tkn)
			filter_set.add(lower_tkn)

	for val in filter_set:
		name = [val]
		wiki_out.writerow(name)

	print('\n'+str(len(filter_set))+'\n')

	print("End of parsing" + file + "\n")

	time.sleep(1)

print('Finished')

wiki_in.close()
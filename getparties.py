import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import difflib

#Getting name and party affiliation.
'''#Open Firefox and go to url

parties = ['Republican', 'Democrat', 'Independent']
masterlist = {}

for party in parties:
	driver = webdriver.Firefox(executable_path=r'C:\geckodriver.exe')

	print('Grabbing {} congressmen'.format(party))

	url = 'https://www.govtrack.us/congress/members/current#current_role_party={}'.format(party)
	print(url)

	driver.get(url)

	SCROLL_PAUSE_TIME = 0.5

	# Get scroll height
	last_height = driver.execute_script("return document.body.scrollHeight")

	while True:
	    # Scroll down to bottom
	    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

	    # Wait to load page
	    time.sleep(SCROLL_PAUSE_TIME)

	    # Calculate new scroll height and compare with last scroll height
	    new_height = driver.execute_script("return document.body.scrollHeight")
	    if new_height == last_height:
	        break
	    last_height = new_height


	members = driver.find_elements_by_class_name('result_item')

	for member in members:
		name = member.find_element_by_tag_name('a').get_attribute('href').split('/')[-2]
		name = name.split('_')
		name = " ".join(name).title()
		if party not in masterlist:
			masterlist[party] = []
		masterlist[party].append(name)
		print(name)

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in masterlist.items() ]))
df.to_csv('parties.csv', encoding='utf-8')'''


#Matching twitter names with real names from parties.
df1 = pd.read_csv('parties.csv', encoding='cp1252')
df2 = pd.read_csv('500tweets.csv', encoding='cp1252')

newnames = []
remove = ['Rep.', 'Rep', 'rep', 'rep.', 'sen', 'sen.', 'Senator', 'U.S.', 'Congresswoman', 'Congressman', 'Sen.']

for column in df2.columns:
	for title in remove:
		if title in column:
			column = column.replace(title, '')
	newnames.append(column)
df2.columns = newnames

df2 = df2.T

democrats = df1['Democrat'].dropna().tolist()
republicans = df1['Republican'].dropna().tolist()
independents = df1['Independent'].dropna().tolist()

for dem in democrats:
	matches = difflib.get_close_matches(dem, df2.index.tolist(), n=1, cutoff=0.6)
	if matches:
		closestMatch = matches[0]
		df2.loc[closestMatch, 'Party'] = 'D'

for rep in republicans:
	matches = difflib.get_close_matches(rep, df2.index.tolist(), n=1, cutoff=0.6)
	if matches:
		closestMatch = matches[0]
		df2.loc[closestMatch, 'Party'] = 'R'

for ind in independents:
	matches = difflib.get_close_matches(ind, df2.index.tolist(), n=1, cutoff=0.6)
	if matches:
		closestMatch = matches[0]
		df2.loc[closestMatch, 'Party'] = 'I'

df2.to_csv('testfinal69.csv', encoding='utf-8')
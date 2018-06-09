import csv
import pandas as pd
import tweepy

#Getting twitter names and handles.
auth = tweepy.OAuthHandler('eHcU05mx45dXC0l3gOLUTAXyg', 'YVdIkqzw91L6dU5hEwcQkyQdcPyThb8gIOlyOdWiYNrMlGSvwJ')
auth.set_access_token('3311641620-NlexlsblK99LbXUA51quO5Tvsq7CQA7KEUkYB3c', '4UqeUPc02BLaS00sXstFJLlo8dfmxIPlCX5Kr9syTAKLa')

api = tweepy.API(auth)

handles = []
realnames = []
for member in tweepy.Cursor(api.list_members, 'cspan', 'members-of-congress').items():
	handles.append(member.screen_name)
	realnames.append(member.name)

df = pd.DataFrame({'Handle': handles, 'Real Names': realnames})
df.to_csv('twitternameshandles.csv', encoding='utf-8')
import csv

csvfile = csv.reader(open('/Users/pengyuzhou/Google Drive/Linkedin_datafile/data.csv',"r"))
total_length = 0
effetive_jobtitle_number = 0
for row in (csvfile):
    if row[3]:
        effetive_jobtitle_number = effetive_jobtitle_number+1
    jobtitle = row[3]
    length = len(jobtitle.split())
    total_length = total_length+length


print('the total length of the jobtitle is {}'.format(total_length))
print('the total number of effective job title is {}'.format(effetive_jobtitle_number))
print('average job title length is {}'.format(float(total_length)/effetive_jobtitle_number))
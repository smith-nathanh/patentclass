"""
Processes the patent publication table's textual fields (description, abstract, claims)
      into a single table with pub number and term

A distribution of the terms is then used to determine
    those that occur too frequently to have meaning (stop words)
    those that occur too infrequently identifying the pub that uses them (overfitting)
"""


import pymysql as db
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer
ps = PorterStemmer()

stop_words=[]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


input_connection = db.connect(user='root', password='patentclass', database='patentclass')
input_cursor = input_connection.cursor()
input_cursor.execute("SELECT pub,dsc,abstr,clms from txt")

output_connection = db.connect(user='root', password='patentclass', database='patentclass')
output_cursor = output_connection.cursor()
output_cursor.execute('truncate terms;')

pubcount=0
for row in input_cursor:
    pubcount += 1
    print(pubcount)
    for i in [1,2,3]:
        stems = [ps.stem(word) for word in tokenizer.tokenize(row[i].lower())]
        filtered_stems = [stem for stem in stems if \
                              not stem in stop_words and \
                              len(stem) > 2 and \
                              not is_number(stem)]
        for w in filtered_stems:
            mute = output_cursor.execute('insert into terms(pub,term,part) values ("' + row[0] +
                                         '","' + w + '",'+str(i)+');')
        mute = output_cursor.execute('commit;')

input_connection.close()
output_connection.close()




stop_words = stopwords.words('english')
stop_words.extend( \
    ['mark', 'letter', 'lowercase', 'uppercase', 'capital', 'font', 'bold', 'claim', 'featur', 'servic', 'name', \
     'word', 'use', 'consist', 'design', 'theme', 'styliz', 'busi', 'provid', 'purpos', 'non', \
     'color', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'grey', 'gray'])
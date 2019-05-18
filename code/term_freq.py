import nltk
import pymysql as db
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')
stop_words.extend( \
    ['mark', 'letter', 'lowercase', 'uppercase', 'capital', 'font', 'bold', 'claim', 'featur', 'servic', 'name', \
     'word', 'use', 'consist', 'design', 'theme', 'styliz', 'busi', 'provid', 'purpos', 'non', \
     'color', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'grey', 'gray'])
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


input_connection = db.connect(user='root', password='b3arclaw', database='patentclass')
input_cursor = input_connection.cursor()
input_cursor.execute("SELECT serial_no,statement_text FROM statement_2017 ORDER BY serial_no")

output_connection = db.connect(user='root', password='', database='trademark')
output_cursor = output_connection.cursor()
output_cursor.execute('truncate tm_words;')
output_cursor.execute('drop index snword on tm_words;')

table_connection = db.connect(user='root', password='', database='trademark')
table_cursor = output_connection.cursor()
table_cursor.execute('truncate tm_statements;')

sn = ""
stmt = ""
counter = 0
for row in input_cursor:
    if row[0] != sn:
        if counter > 0:
            print(counter)
            stems = [ps.stem(word) for word in tokenizer.tokenize(stmt.lower())]
            filtered_statement = [w for w in stems if \
                                  not w in stop_words and \
                                  len(w) > 2 and \
                                  not is_number(w)]
            sql = 'insert into tm_statements(serial_no,statement) values ("' + sn + '","' + stmt.replace('"',
                                                                                                         '').replace(
                "'", "") + '");'
            table_cursor.execute(sql)
            table_cursor.execute('commit;')
            for w in filtered_statement:
                output_cursor.execute('insert into tm_words(serial_no,word) values ("' + sn + '","' + w + '");')
            output_cursor.execute('commit;')
            print("Committed!")
            stmt = row[1]
        sn = row[0]
    else:
        counter = counter + 1
        stmt = stmt + " " + row[1]

output_cursor.execute('create index snword on tm_words(serial_no,word);')
input_connection.close()
output_connection.close()

import jsonlines
import re
from textblob import TextBlob
from collections import Counter

def tprint(obj):
    print(f"{type(obj)} : {obj}")


def sort_file(file):
    '''return number of positive and negative comments based on TextBlob sentiment analysis'''
    # with jsonlines.open("/vadoe/vadoe/vadoe/townhall_2021-01-14T02-05-51.json") as reader:
    with jsonlines.open(file, mode='r') as reader:
        # Confirm type
        tprint(reader)

        # Build iterator
        _doc = iter(reader)
        i = 0
        pos = 0
        neg = 0
        posl = []
        negl = []

        while i<25:
            _line = next(_doc)
            tprint(_line)
            if _line['sentiment'] == 'pos':
                pos = pos + 1
                posl.append(_line['comment'])
            elif _line['sentiment'] == 'neg':
                neg = neg + 1
                negl.append(_line['comment'])
            i=i+1

        print(f'{pos} positive and {neg} negative comments')
            # tst = TextBlob(obj['comment'])
            # tst.sentiment

def process_file(file):
    '''Find Smythers posts'''
    with jsonlines.open(file, mode='r') as reader:
        _doc = iter(reader)
        _list = []
        for item in _doc:
                try:
                    if item['author'][0] == 'Smythers': 
                        _list.append(item['content'][0])
                except KeyError:
                    continue
    return(_list)

def write_file(file, data:object):
    '''Write data to file'''
    with jsonlines.open(file, mode='w') as writer:
        for each in data:
            writer.write(each)
    print('write successful')

def clean_text(text:str):
    s1 = remove_html(text)
    s2 = remove_http(s1)
    return s2

def remove_html(text:str):
    '''Remove html tags from string'''
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_http(text:str):
    '''Remove URLs from string'''
    return re.sub(r'http\S+','', text)

def get_nouns(text:str):
    blob = TextBlob(text)
    # check nouns? or no
    return blob.tags

vadoe = '/vadoe/vadoe/vadoe/townhall_2021-01-14T02-05-51.json'
vadoe_p = '/vadoe/vadoe/vadoe/townhall_2021-01-14T05-11-55.json'
dlr = '/vadoe/vadoe/vadoe/dlr.json'

smythers_pc = '/vadoe/vadoe/vadoe/smythers.json'
write_to = '/vadoe/vadoe/vadoe/nouns.json'

# processed_file(file)
smythers_posts = process_file(dlr)
# cleaned = []
# for each in smythers:
    # cleaned.append(clean_text(each))
cleaned = [clean_text(each) for each in smythers_posts]
nouns = []
for x in cleaned:
    _list = get_nouns(x)
    for y in _list:
        nouns.append(y)
    # nouns.append(x for x in [get_nouns())
sortedNouns = Counter(nouns)
nouns = []
for k, v in sortedNouns.items():
    if v > 2: 
        _d = (k, v)
        nouns.append(_d)
print(nouns)
write_file(write_to, nouns)
import sys
import re

keywords = [
'oxycodone',
'methadone',
'morphine',
'tramadol',
'hydrocodone',
'buprenorphine',
'naloxone',
'diazepam',
'alprazolam',
'clonazepam',
'lorazepam',
'olanzapine',
'risperidone',
'aripiprazole',
'asenapine',
'quetiapine',
'amphetamine mixed salts',
'lisdexamfetamine',
'methylphenidate',
'gabapentin',
'pregabalin',
]

regex_filter = re.compile(r'|'.join(keywords))

def filter(text):
    if regex_filter.search(text) != None:
        return True
    return False


if __name__ == '__main__':
    for line in sys.stdin:
        text = line.strip()
        if filter(text):
            print(text)

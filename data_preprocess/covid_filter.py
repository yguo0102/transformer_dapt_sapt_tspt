import re
import sys

filters = ['\bcovid\b', 'coronavirus']
regex_filter = re.compile(r'|'.join(filters))

def filter(text):
    text = text.lower()
    if regex_filter.search(text) != None:
        return True
    return False


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if filter(line):
            print(line)

#!/usr/bin/python
import sys
import csv
import re
from itertools import permutations
import pdb

def format_keywords_pattern(arr):
    '''
    Create the regular expression that can match multiple keywords in any order.
    For example, to match the text including 'A', 'B', and 'C', the regular expression would be
    '(A.*B.*C)|(A.*C.*B)|(B.*A.*C)|(B.*C.*A)|(C.*A.*B)|(C.*B.*A)'
    '''
    perm = permutations([i for i in range(len(arr))])
    terms = []
    for order in perm:
        new_arr = []
        for i in order:
            new_arr.append(arr[i])
        terms.append('({})'.format('.*'.join(new_arr)))
    return '({})'.format('|'.join(terms))


filters = [
    format_keywords_pattern(['#?breast', '#?cancer', '#?survivor']),
    format_keywords_pattern(['#?breastcancer', '#?survivor']),
    format_keywords_pattern(['#?tamoxifen', '#?cancer']),
    format_keywords_pattern(['#?tamoxifen', '#?survivor']),
    format_keywords_pattern(['(my|i|me)', '#?breast', '#?cancer']),
    format_keywords_pattern(['(my|i|me)', '#?breastcancer']),
    format_keywords_pattern(['(my|i|me)', '#?tamoxifen'])
]
regex_filter = re.compile(r'|'.join(filters))
rt_filter = re.compile(r'rt <user>')


def filter(text):
    if rt_filter.search(text) == None and regex_filter.search(text) != None:
        return True
    return False

if __name__ == '__main__':
    #text = 'PURPOSE'
    #print(filter(text))
    for line in sys.stdin:
        line = line.strip()
        if filter(line):
            print(line)

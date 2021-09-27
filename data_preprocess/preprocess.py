"""
The script is a Python version of the Ruby script for Twitter data preprocessing
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


eyes = r"[8:=;]"
nose = r"['`\-]?"
re_patterns = {
'url': re.compile(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"),
'user': re.compile(r"@\w+"),
'smile' : re.compile(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes)),
'lolface' : re.compile(r"{}{}p+".format(eyes, nose)),
'sadface' : re.compile(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes)),
'neutralface' : re.compile(r"{}{}[\/|l*]".format(eyes, nose)),
'slash' : re.compile(r"/"),
'heart' : re.compile(r"<3"),
'number' : re.compile(r"[-+]?[.\d]*[\d]+[:,.\d]*"),
'hashtag' : re.compile(r"#\S+"),
'repeat' : re.compile(r"([!?.]){2,}"),
'elong' : re.compile(r"\b(\S*?)(.)\2{2,}\b"),
'allcaps' : re.compile(r"([A-Z]){2,}"),
}

def tokenize(text):
    # Different regex parts for smiley faces

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_patterns['url'].sub("<url>", text)
    text = re_patterns['user'].sub("<user>", text)
    text = re_patterns['smile'].sub("<smile>", text)
    text = re_patterns['lolface'].sub("<lolface>", text)
    text = re_patterns['sadface'].sub("<sadface>", text)
    text = re_patterns['neutralface'].sub("<neutralface>", text)
    text = re_patterns['slash'].sub(" / ", text)
    text = re_patterns['heart'].sub("<heart>", text)
    text = re_patterns['number'].sub("<number>", text)
    text = re_patterns['hashtag'].sub(hashtag, text)
    text = re_patterns['repeat'].sub(r"\1 <repeat>", text)
    text = re_patterns['elong'].sub(r"\1\2 <elong>", text)

    text = re_patterns['allcaps'].sub(allcaps, text)

    return text.lower()


def process_tweets(text):
    return tokenize(text)

def process_reddit(text):
    text = tokenize(text)
    if '[removed]' in text or '[deleted]' in text:
        return ''
    return text

def is_long(text):
    arr = text.strip().split()
    if len(arr) > 5:
        return True
    else:
        return False


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        line = tokenize(line)
        print(line)



#!/usr/bin/python3

from nltk.stem.snowball import SnowballStemmer
import string, re

from nltk.stem import *



def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    print(all_text)

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        # print(len(string.punctuation))
        # text_string = content[1].translate(None, string.punctuation)
        text_string = re.sub('[%s]' % re.escape(string.punctuation), '', content[1])
        ### project part 2: comment out the line below
        # words = text_string

        orig_words = text_string.split()
        # print(len(orig_words))


        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words = ""
        stemmer = SnowballStemmer("english")
        for w in orig_words:
            words += " " + stemmer.stem(w)
            # append(stemmer.stem(w)) if stemmer.stem(w) not in words
    return words



def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

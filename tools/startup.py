#!/usr/bin/python3

print("")
print("checking for nltk")
try:
    import nltk
except ImportError:
    print("you should install nltk before continuing")

print("checking for numpy")
try:
    import numpy
except ImportError:
    print("you should install numpy before continuing")

print("checking for scipy")
try:
    import scipy
except:
    print("you should install scipy before continuing")

print("checking for sklearn")
try:
    import sklearn
except:
    print("you should install sklearn before continuing")

print("")
print("downloading the Enron dataset (this may take a while)")
print("to check on progress, you can cd up one level, then execute <ls -lthr>")
print("Enron dataset should be last item on the list, along with its current size")
print("download will complete at about 423 MB")
import urllib
import ssl
import shutil
ssl._create_default_https_context = ssl._create_unverified_context
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
# urllib.request.urlopen(url, filename="../enron_mail_20150507.tgz")
file_name = "enron_mail_20150507.tgz"
with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
# with urllib.request.urlopen(url) as response:
#     with gzip.GzipFile(fileobj=response) as uncompressed:
#         file_header = uncompressed.read(64) # a `bytes` object
        # Or do anything shown above using `uncompressed` instead of `response`.

print("download complete!")


print("")
print("unzipping Enron dataset (this may take a while)")
import tarfile
import os
# os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
tfile.extractall(".")

print("you're ready to go!")

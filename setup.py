import setuptools
with open ("README.md" , "r" , encoding="utf-8") as f :
    long_description = f.read()

version = "0.0.0"
RepoName = "image-classification"
Author_User_Name = "tahasalmani"
SRC_REPO = "cnnclassifier"
Author_email = "the.taha.salmani@gmail.com"

setuptools.setup(
    name = SRC_REPO ,
    version = version ,
    author = Author_User_Name ,
    author_email = Author_email ,
    description = "image classifier" ,
    url = f"https://github.com/{Author_User_Name}/{RepoName}"





)


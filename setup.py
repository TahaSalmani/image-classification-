import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

version = "0.0.0"
RepoName = "image-classification-"
Author_User_Name = "tahasalmani"
SRC_REPO = "cnnclassifier"
Author_email = "the.taha.salmani@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=version,
    author=Author_User_Name,
    author_email=Author_email,
    description="image classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{Author_User_Name}/{RepoName}",
    # src-layout: without these, pip install -e . discovers nothing importable
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)

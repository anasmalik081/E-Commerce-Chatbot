from setuptools import find_packages, setup

setup(
    name="Ecommercebot",
    version="0.0.1",
    author="Anas",
    author_email="anasmalik081@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain ','langchain-google-genai','datasets','pypdf','python-dotenv','flask']
)
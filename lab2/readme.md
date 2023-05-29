[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)


1. [Launch Jenkins](https://www.jenkins.io/doc/tutorials/build-a-python-app-with-pyinstaller/#on-macos-and-linux)
2. [Init Jenkins](https://www.jenkins.io/doc/tutorials/build-a-python-app-with-pyinstaller/#setup-wizard)
3. create new pipeline with next params:
   1. Definition: Pipeline script from SCM
   2. SCM: Git
   3. Repository URL: https://github.com/brain4power/mlops_labs
   4. Branch Specifier: */main
   5. Script Path: lab2/Jenkinsfile
4. Save and run pipeline
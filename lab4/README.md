# Lab4

## How to start using:
1. install requirements:
```shell
pip install -r lab4/requirements.txt
```
2. Pull files from remote disk:
```shell
dvc pull -r mydisc
```
Browser will open, login with your google acc and give all permissions to DVC.
Files should appear at `lab4/data` directory
## How to use
1. Create new files, change old at lab4/data directory
2. Add new file to dvc
```shell
dvc add lab4/data/new_file.csv
```
3. change files and commit changes:
```shell
dvc commit
```
4. Push changes to remote disk
```shell
dvc push -r mydisc
```
5. remove file
```shell
rm -rf lab4/data/new_file.csv
dvc commit
dvc push -r mydisc
```
## Additional info

[Google drive link](https://drive.google.com/drive/folders/1cBq5fewRBhq_HeDtDPgH4mbabiqSRbBo)

Drive ID: 1cBq5fewRBhq_HeDtDPgH4mbabiqSRbBo

Download csv (only train.csv): [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data?select=train.csv)

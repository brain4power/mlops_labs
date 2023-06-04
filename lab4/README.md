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
# Dataset transformations
```shell
cd lab4
```
- Transform "Age" values to categories and add new column
```shell
python transform_dataset.py \
    --source_file_name titanic.csv \
    --actions age_to_categories \
    --column_names Age \
    --rewrite
```
- Fill nan values at "Cabin" column by "not_indicated" value
```shell
python transform_dataset.py \
    --source_file_name titanic.csv \
    --actions fill_na_values \
    --column_names Cabin \
    --fill_na_action by_text \
    --fill_na_text not_indicated \
    --rewrite
```
- Fill nan Age column values by mean value
```shell
python transform_dataset.py \
    --source_file_name titanic.csv \
    --actions fill_na_values \
    --column_names Age \
    --fill_na_action mean \
    --rewrite
```
- Make one-hot-encoding for columns Sex, Embarked, age_class
```shell
python transform_dataset.py \
    --source_file_name titanic.csv \
    --actions one_hot_encoding \
    --column_names Sex,Embarked,age_class \
    --rewrite
```
- Make one-hot-encoding for columns Sex, Embarked, age_class, Save to new file with name titanic_ohe.csv
```shell
python transform_dataset.py \
    --source_file_name titanic.csv \
    --actions one_hot_encoding \
    --column_names Sex,Embarked,age_class \
    --result_name titanic_ohe
```
## Switch between different dataset versions
Show git log
```shell
git log --oneline
```
```shell
96fcf5f feat: dataset: Fill nan Age column values by mean value
aab1490 feat: dataset: Fill nan values at "Cabin" column by "not_indicated" value
eccd73a feat: dataset: Transform "Age" values to categories and add new column
999ac65 feat: default titanic dataset
```
Find the hash of the desired commit and checkout to it
```shell
git checkout 999ac65
```
Pull data changes from dvc storage
```shell
dvc pull -r mydisc
```
## Additional info

[Google drive link](https://drive.google.com/drive/folders/1cBq5fewRBhq_HeDtDPgH4mbabiqSRbBo)

Drive ID: 1cBq5fewRBhq_HeDtDPgH4mbabiqSRbBo

Download csv (only train.csv): [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data?select=train.csv)

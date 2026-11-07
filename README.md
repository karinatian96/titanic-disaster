# titanic-disaster

Logistic regression models (Python and R) that predict passenger survival on the Titanic. Each model runs in its own Docker container.

## Project structure

- `src/data/` - put `train.csv` and `test.csv` here (not tracked by git)
- `src/python/` - `Dockerfile`, `requirements.txt`, `main.py`
- `src/r/` - `Dockerfile`, `install_packages.R`, `main.R`

## 1. Get the data

1. Go to https://www.kaggle.com/competitions/titanic/data (join the competition and accept the rules).
2. Download `train.csv` and `test.csv`.
3. Put both files in `src/data/`.

The data is not included in this repo because it is ignored by `.gitignore`.

## 2. Run the Python container

From the repository root:

    docker build -t titanic-python src/python
    docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python

## 3. Run the R container

From the repository root:

    docker build -t titanic-r src/r
    docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r

The first R build may take several minutes because packages are compiled from source.

## What the scripts do

- Load `train.csv` and report missing values
- Fill missing Age/Fare with the median and Embarked with the mode
- Encode Sex and Embarked, add FamilySize, drop Cabin/Name/Ticket
- Train a logistic regression model and report training accuracy
- Predict survival on `test.csv`
- `test.csv` has no `Survived` labels, so accuracy is reported on a 20% hold-out split of `train.csv`

## Results

| Metric | Python | R |
|---|---|---|
| Training accuracy | 0.8076 | 0.8062 |
| Hold-out accuracy | 0.8045 | 0.7598 |
| Predicted survivors in test set | 153 / 418 | 156 / 418 |

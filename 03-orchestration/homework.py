import pickle
from datetime import datetime

from pathlib import Path
from typing import Optional, Any

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import prefect


@prefect.task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@prefect.task
def prepare_features(df, categorical, train=True):
    logger = prefect.get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info("The mean duration of training is %f", mean_duration)
    else:
        logger.info("The mean duration of validation is %f", mean_duration)

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@prefect.task
def train_model(df, categorical):
    logger = prefect.get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is %s", X_train.shape)
    logger.info(f"The DictVectorizer has %d features", len(dv.feature_names_))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: %f", mse)
    return lr, dv


@prefect.task
def run_model(df, categorical, dv, lr):
    logger = prefect.get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: %f", mse)
    return


def _string_to_date(date: str) -> datetime.date:
    return datetime.strptime(date, "%Y-%m-%d").date()


def _date_to_string(date: datetime.date) -> str:
    return date.strftime("%Y-%m")


def _ensure_data_exists(path: Path):
    if path.exists():
        return

    response = requests.get(f"https://nyc-tlc.s3.amazonaws.com/trip+data/{path.stem}.parquet", verify=False)
    response.raise_for_status()

    with path.open("wb") as file:
        file.write(response.content)


@prefect.task
def get_paths(date: Optional[str] = None, date_folder: Path = Path("./data")):
    logger = prefect.get_run_logger()

    if date is None:
        date = datetime.now().date()
    else:
        date = _string_to_date(date)

    train_date = date - relativedelta(months=2)
    val_date = date - relativedelta(months=1)

    train_path = date_folder / f"fhv_tripdata_{_date_to_string(train_date)}.parquet"
    val_path = date_folder / f"fhv_tripdata_{_date_to_string(val_date)}.parquet"

    logger.info("Train path %s", train_path)
    logger.info("Val path %s", val_path)

    _ensure_data_exists(train_path)
    _ensure_data_exists(val_path)

    return train_path, val_path


def _save_to_bin(obj: Any, path: Path):
    with path.open('wb') as file:
        pickle.dump(obj, file)


@prefect.flow
def main(date: Optional[str] = None):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    _save_to_bin(lr, Path("./artifacts") / f"model-{date}.bin")
    _save_to_bin(dv, Path("./artifacts") / f"dv-{date}.bin")


DeploymentSpec(
    name="cron-schedule-deployment",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
    ),
    flow_runner=SubprocessFlowRunner(),
)


# main(date="2021-08-15")

# imports
import pandas as pd
from data import get_data, clean_data, get_test_data
from utils import compute_rmse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from encoders import TimeFeaturesEncoder, DistanceTransformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])

        self.pipeline = pipe
        return pipe

    def run(self):
        pass

    def run(self):
        """set and train the pipeline"""
        return self.pipeline.fit(X, y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

if __name__ == "__main__":
    # get data
    # clean data
    print(f'Getting data...')
    df = clean_data(get_data())

    # set X and y
    print(f'Setting X und y...')

    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f'Set Pipeline...')
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()

    # train
    print(f'Run...')
    trainer.run()

    # evaluate
    print(f'Evaluate...')
    # df_test = clean_data(get_test_data())
    # set X_test and y_test
    res = trainer.evaluate(X_test, y_test)
    # print(res)

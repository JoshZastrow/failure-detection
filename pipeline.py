from metaflow import Metaflow, FlowSpec, step, Parameter, Flow, current
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import numpy as np


class Model(FlowSpec):
    """
    An end-to-end pipeline for preprocessing, modeling, evaluating and serving
    predictions for predicting device failure in a remote area.
    """

    mode = Parameter('mode',
                      help='Specify mode as either "train" or "serve"',
                      default='train')
    
    fpath = Parameter('fpath',
                      help='Path to data file.',
                      default='./data/predict_failure.csv')

    lag = Parameter('lag',
                    help='Number of days back to compute change in sensor values',
                    default=5)

    threshold = Parameter('threshold',
                        help="""
                        The percentage of outliers present in a rolling window
                        to be considered a failing device. Value between 0 and 1""",
                        default=0.5
                        )

    @step
    def start(self):
        """
        Ingest the dataset and instantiate a model (in case of training) or load in a model
        (in case of serving)
        """
        if not os.path.exists(self.fpath):
            raise FileNotFoundError(f'Cannot find {self.fpath} from {os.getcwd()}')

        self.df = pd.read_csv(self.fpath)

        if self.mode == 'serve':
            if os.path.exists('./data/model.gz'):
                self.model = joblib.load('./data/model.gz')
            else:
                raise FileNotFoundError('No model file. Please run pipeline in train mode.')
        else:
            self.model = RandomForestClassifier(
                bootstrap=False, 
                criterion="entropy", 
                max_features=0.15, 
                min_samples_leaf=1, 
                min_samples_split=15, 
                n_estimators=100)

        self.next(self.clean)


    @step
    def clean(self):
        """Convert date to datetime, sort by date/device and reset the index"""
        self.df = (
            self.df
            .assign(date=lambda d: d.date.astype('datetime64[ns]'))
            .sort_values(['date', 'device'])
            .reset_index(drop=True)
        )
        self.next(self.calc_lagging_trend)


    @step
    def calc_lagging_trend(self):
        """Calculate a lagging regression on a few of the attributes"""

        cols = ['attribute2', 'attribute4', 'attribute7', 'attribute8']
        df = self.df.copy()

        lag_features = (
            df
            .groupby('device')
            .apply(lambda df: 
                df
                .loc[:, cols]
                .transform(lambda x: ((x - x.shift(self.lag)) / self.lag).fillna(0))
                )
            .rename(columns={c: c + '_lag' for c in cols})
            .sort_index()
        )

        self.df = df.merge(lag_features, left_index=True, right_index=True)
        
        self.next(self.roll_normalization)

    @step
    def roll_normalization(self):
        """Normalize daily values and lags against a rolling 10 day statistic.

        1) Calculate a rolling statistic across all devices over a period of 10 days
        2) Group by date and keep the last row of each day.
        3) Normalize each devices daily readings against the rolling statistic for a given day.
        """
        df = self.df.copy()

        rolling_statistics = (
            df
            .drop(['failure', 'device'], axis=1)
            .set_index('date')
            .rolling('10d', min_periods=1)
            .agg(['mean', 'std'])
            .groupby('date')
            .apply(lambda x: x.tail(1)) 
            .reset_index(level=0, drop=True)
            .pipe(lambda df: df.set_axis(df.columns.to_flat_index().map('_'.join), axis=1))
        )
        
        def normalize(col, ref_df):
            '''Normalize a column using the tacked on rolling statistics'''
            return ((col - ref_df[col.name + '_mean']) / (ref_df[col.name + '_std'])).fillna(0)

        cols = df.columns[3:]
            
        self.df = (
            df
            .merge(rolling_statistics, left_on='date', right_index=True)
            .pipe(lambda merged: 
                pd.concat([
                    merged.iloc[:, :3],
                    merged.loc[:, cols].transform(normalize, ref_df=merged),
                ], axis=1)
            )
        )

        self.next(self.split)

    @step
    def split(self):
        """
        Perform an 80/20 split across devices, stratified by major and minor (failed) class.
        Use latest readings per device for training and evaluating.
        """
        def label_most_recent_readings(df, device_list):
            return (
                df.copy()
                .loc[df.device.isin(device_list)]
                .groupby('device')
                .apply(lambda sub: 
                    sub
                    .sort_values('date')
                    .iloc[-20:, :]
                    .assign(failure=lambda d: d.failure.max())
                    )
                .reset_index(level=0, drop=True)
            )

        if self.mode == 'train':
            df = self.df.copy()

            failure = df[df.failure==1].device.unique()
            working = df[(df.failure==0) & ~(df.device.isin(failure))].device.unique()
            
            maj_train_devices, maj_valid_devices = train_test_split(working, test_size=0.2, random_state=7)
            min_train_devices, min_valid_devices = train_test_split(failure, test_size=0.2, random_state=7)
            
            maj_train = df.pipe(label_most_recent_readings, maj_train_devices)
            min_train = df.pipe(label_most_recent_readings, min_train_devices)
            maj_valid = df.pipe(label_most_recent_readings, maj_valid_devices)
            min_valid = df.pipe(label_most_recent_readings, min_valid_devices)
            
            train = pd.concat([
                maj_train,
                min_train
            ], axis=0)
            
            valid = pd.concat([
                maj_valid,
                min_valid
            ], axis=0)
            
            print('Training | Number of devices per failure status')
            print(train.groupby('failure').device.nunique())
            print()
            print('Validation | Number of devices per failure status')
            print(valid.groupby('failure').device.nunique())
            
            self.X_train = train.drop(['failure', 'device', 'date'], axis=1)
            self.y_train = train.loc[:, 'failure']
            self.X_valid = valid.drop(['failure', 'device', 'date'], axis=1)
            self.y_valid = valid.loc[:, 'failure']

        self.X_serve = self.df.copy()
        for c in ['failure', 'device', 'date']:
            if c in self.X_serve:
                del self.X_serve[c]

        self.next(self.train_evaluate)

    @step
    def train_evaluate(self):
        """
        Train a random forest classifier on ~80 failed devices and ~850 working devices
        Evaluate model against remaining devices. 
        
        An accurate prediction is when the number of consecutive failure predictions 
        for a device over the evaluation period exceeds a certain threshold. This is to
        normalize against outliers / single day noise in sensor readings.
    
        """
        if self.mode == 'train':
            print('Training Model...')
            X_train = self.X_train.copy()
            y_train = self.y_train.copy()
            X_valid = self.X_valid.copy()
            y_valid = self.y_valid.copy()

            thresh = self.threshold
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, './data/model.gz')

            assert y_valid.index.equals(X_valid.index)
            y_pred = pd.Series(self.model.predict(X_valid), index=X_valid.index)

            # Outlier Models (1 inlier, -1 outlier)
            if -1 in y_pred:
                y_pred = y_pred.replace(1, 0).replace(-1, 1)
    
            results = (
                    self.df
                    .loc[y_pred.index, :]
                    .assign(prediction=y_pred.values)
                    .groupby('device')
                    .agg(
                        detected=('prediction', lambda x: (x.sum() / x.count()) > thresh), 
                        actual=('failure', 'max')
                    )
            )

            self.confusion_matrix = metrics.confusion_matrix(
                y_pred=results.detected, 
                y_true=results.actual
                )
            self.precision = metrics.precision_score(
                y_pred=results.detected,
                y_true=results.actual
                )
            self.recall = metrics.recall_score(
                y_pred=results.detected,
                y_true=results.actual
                )

            self.results = results

        self.next(self.serve_predict)


    @step
    def serve_predict(self):
        """
        Serve predictions for the entire dataset. Dump the results to an output table.
        """
        print('Serving Predictions...')
        y_pred = self.model.predict(self.X_serve)

        self.output = self.df.copy()
        self.output['predictions'] = None

        self.output.loc[self.X_serve.index, 'predictions'] = y_pred
        
        self.dest_path = f'./data/serve_predictions_{current.run_id}.csv'

        print(f'Output predictions dumped to {self.dest_path}')
        self.output.to_csv(
            self.dest_path, 
            index=True, 
            index_label='index')

        self.next(self.end)

    @step
    def end(self):
        print('Completed!')

if __name__ == '__main__':
    Model()

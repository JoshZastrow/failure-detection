import pandas as pd
import numpy as np

def clean(df):
    return (
        df.copy()
        .assign(date=lambda d: d.date.astype('datetime64[ns]'))
        .sort_values(['device', 'date'])
    )


def feat_days_prior(df):
    def days_prior(df):
        return (
            df
            .date
            .transform(lambda s: s - s.max())
            .transform(lambda s: s.days)
            .rename('days_prior')
        )
    
    df = (
        df
        .pipe(groupby_device)
        .apply(days_prior)
        .reset_index(drop=True)
        .pipe(pd.DataFrame)
    )
    return df

def feat_moving_variance(df):
    def moving_variance(df):
        return (
            df
            .set_index('date')
            .rolling(5)
            .var()
            .bfill()
        )
    
    return (
        df
        .pipe(groupby_device)
        .apply(moving_variance)
        .reset_index()
        .drop('failure', axis=1)
    )

def feat_device_age(df):
    def device_age(df):
        return (
            df
            .date
            .transform(lambda s: s - s.min())
            .transform(lambda s: s.days)
            .rename('age')
        )
    
    df = (
        df
        .pipe(groupby_device)
        .apply(device_age)
        .reset_index(drop=True)
        .pipe(pd.DataFrame)
    )
    return df


def feat_device_lifetime_change(df):
    df = (
        df.copy()
        .pipe(groupby_device)
        .transform(lambda s: s - s.iloc[0])
        .iloc[:, 3:]
    )
    return df


def feat_z_score_at_age(df):
    def z_score(s):
        return (s - s.mean()) / s.std(ddof=0)

    df = (df.copy()
        .groupby('age')
        .apply(lambda d: 
               d.assign(
                   attribute1=z_score(d.attribute1),
                   attribute2=z_score(d.attribute2),
                   attribute4=z_score(d.attribute4),
                   attribute7=z_score(d.attribute7),
                   attribute8=z_score(d.attribute8),
               ))
        .reset_index(drop=True)  
        .loc[:, ['attribute2', 'attribute4', 'attribute7', 'attribute8']]
    )
    
    return df


def feature_lag(df):
    df = (
        df.copy()
        .assign(date=lambda d: d.date.astype('datetime64[ns]'))
        .groupby('device')
        .apply(lambda df: 
               df
               .set_index('date')
               .assign(delta_1=(df.attribute1 - df.attribute2.shift(1))).bfill()
              )
        .reset_index(level=0, drop=True)
    )
    return df

def pad_history(df):
    padded_df = pd.DataFrame(np.nan, columns=df.columns, index=range(50))
    padded_df.iloc[-df.shape[0]:, :] = df.iloc[-50:, :].values
    padded_df = padded_df.bfill()
    padded_df['failure'] = padded_df.iloc[-1,2]
    return padded_df


def feat_backfill_failure(df):
    def backfill_failure(df):
        failed = df.failure.iloc[-1]
        if failed:
            df.failure.iloc[-5:] = failed
                
        return df.failure
    
    return (
        df.copy()
        .sort_values(['device', 'date'])
        .groupby('device')
        .apply(lambda df: df.assign(failure=backfill_failure))
        .reset_index(level=0, drop=True)
    )

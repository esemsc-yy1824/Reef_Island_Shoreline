from sklearn.pipeline import Pipeline
from preprocessing.transformers import (
    DateTransformer,
    WindTransformer,
    MonthTransformer,
    NumericScaler
)


def build_pipeline():
    cols_to_scale = [
        'date_ordinal',
        'wind_speed',
        'significant_height_of_combined_wind_waves_and_swell',
        'mean_wave_period',
        'slope_m_per_m',
        'NDVI',
        'reef_width_m',
        'reef_crest_elevation_m',
        'reef_flat_mean_depth_m',
        'reef_slope_deg',
    ]

    pipeline = Pipeline(steps=[
        ('date', DateTransformer(date_column='date')),
        ('wind', WindTransformer()),
        ('month', MonthTransformer(month_col='month')),
        ('scaler', NumericScaler(cols_to_scale=cols_to_scale))
    ])

    return pipeline
import dask.dataframe as dd

cols = ['down',
        'posteam',
        'yardline_100',
        'game_date',
        'game_seconds_remaining',
        'ydstogo',
        'score_differential',
        'fourth_down_converted',
        'fourth_down_failed',
        'field_goal_attempt',
        'punt_attempt'
        ]
df = dd.read_csv('nfl_data.csv', usecols=cols, assume_missing=True)
df['game_date'] = dd.to_datetime(df['game_date'])
df_filtered = df.loc[df['down'] == 4]

# Get unique years
years = df_filtered['game_date'].dt.year.unique().compute()

# Save each year to a separate CSV file
for year in years:
    filename = f"../data/{year}.csv"
    df_year = df_filtered.loc[df_filtered['game_date'].dt.year == year]
    df_year.to_csv(filename, index=False, single_file=True)

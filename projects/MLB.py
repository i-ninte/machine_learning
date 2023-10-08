def col_list_sum(df, col_list, weights=None):
    col_df = df[col_list]
    if weights is not None:
        col_df = col_df.multiply(weights)
    return col_df.sum(axis=1)


mlb_df['BA']=mlb_df['H']/mlb_df['AB']

other_hits= col_list_sum(mlb_df,['2B','3B','HR'])
mlb_df['1B']= mlb_df['H'] - other_hits

import pandas as pd
import os

stocks = ['$AAPL', '$AXP', '$BA', '$CAT', '$CSCO', '$CVX', '$DD', '$DIS', '$GS', '$HD', '$IBM', '$INTC', '$JNJ', '$JPM', '$KO',
                  '$MCD', '$MMM', '$MRK', '$MSFT', '$NKE', 'PFE', '$PG', '$RTX', '$TRV', '$UNH', '$V', '$VZ', '$WBA', '$WMT', '$XOM']

for stock in stocks:
    path = "twitter_prob_data/"+stock+"_twitter"+".csv"
    df = pd.read_csv(path)
    # print(df['date'].head())
    tweets_prob_action_array = df.loc[df['date']==20170103]['action_prob'].values
    if (len(tweets_prob_action_array) == 0):
        tweet_prob_action = 0
    else:
        tweet_prob_action = tweets_prob_action_array[0]
    print(tweet_prob_action)
    print(type(tweet_prob_action))

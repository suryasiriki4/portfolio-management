import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# sentiment analysis modules
import os.path
from datetime import datetime, timedelta
import flair
import regex as re
matplotlib.use('Agg')

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = 30
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False             
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        #self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             (1- TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        # print('available_amount:{}'.format(available_amount))

        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1+ TRANSACTION_FEE_PERCENT)

        self.state[index+STOCK_DIM+1] += min(available_amount, action)

        self.cost+=self.state[index+1]*min(available_amount, action)* \
                          TRANSACTION_FEE_PERCENT
        self.trades+=1

    def modifyActions(self, actions):
        new_actions = [0]*30
        stocks = ['$AAPL', '$AXP', '$BA', '$CAT', '$CSCO', '$CVX', '$DD', '$DIS', '$GS', '$HD', '$IBM', '$INTC', '$JNJ', '$JPM' '$KO',
                  '$MCD', '$MMM', '$MRK', '$MSFT', '$NKE', 'PFE', '$PG', '$RTX', '$TRV', '$UNH', '$V', '$VZ', '$WBA', '$WMT', '$XOM']
        day = str(self.data.iat[0, 0])
        # day = '201701s04'
        # print(day)
        curr_date = pd.to_datetime(day, format="%Y%m%d")
        year = day[0:4]
        i = 0
        # date = day[0:4] + "-" + day[4:6] + "-" + day[6:]
        for stock in stocks:
            # print(i)
            path = "tweets_data/"+stock+"_"+year+"_nlpData.csv"
            if os.path.isfile(path):
                recent_dates = []
                # print(path)
                df = pd.read_csv(path)
                df.sort_values(by=['Timestamp'])
                dates = []
                for columnData in df['Timestamp']:
                    # print('Colunm Name : ', columnName)
                    # print(columnData[0:10])
                    dates.append(columnData[0:10].replace('-', ''))
                df['dates'] = dates
                # print("curr")
                # print(curr_date)
                for past in range(1, 4):
                    recent_dates.append(
                        ((curr_date - timedelta(past)).strftime("%Y/%m/%d")).replace('/', ''))
                # print("recent")
                # print(recent_dates)
                # print(df.dates)
                tweets = pd.DataFrame()
                tweets['text'] = df[(df.dates >= recent_dates[2]) & (
                    df.dates <= recent_dates[0])].Text
                # print(tweets.head())
                sentiment_model = flair.models.TextClassifier.load(
                    'en-sentiment')
                probs = []
                sentiments = []
                for tweet in tweets['text'].to_list():

                    whitespace = re.compile(r"\s+")
                    web_address = re.compile(
                        r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
                    # tesla = re.compile(r"(?i)@Tesla(?=\b)")
                    user = re.compile(r"(?i)@[a-z0-9_]+")

                    # we then use the sub method to replace anything matching
                    tweet = whitespace.sub(' ', tweet)
                    tweet = web_address.sub('', tweet)
                    # tweet = tesla.sub('Tesla', tweet)
                    tweet = user.sub('', tweet)

                    sentence = flair.data.Sentence(tweet)
                    sentiment_model.predict(sentence)
                    # extract sentiment prediction
                    # numerical score 0-1
                    probs.append(sentence.labels[0].score)
                    # 'POSITIVE' or 'NEGATIVE'
                    if(sentence.labels[0].value == 'POSITIVE'):
                        sentiments.append(1)
                    else:
                        sentiments.append(-1)
                    # sentiments.append(sentence.labels[0].value)

                # add probability and sentiment predictions to tweets dataframe
                # print(probs)
                # print(sentiments)
                product = [probs[i]*sentiments[i] for i in range(len(probs))]
                tweets['probability'] = probs
                tweets['sentiment'] = sentiments
                tweets['product'] = product

                # print(tweets.head())
                # print(tweets['product'].mean())
                # # return tweets['product'].mean()
                # print("RL "+str(actions[i]))
                # print("NLP "+str(tweets['product'].mean()*HMAX_NORMALIZE))
                new_actions[i] = 0.7 * actions[i] + 0.3 * tweets['product'].mean() * HMAX_NORMALIZE
                # print("NA "+str(new_actions[i]))
                # i = i + 1
                # new_df  = df.loc[df['date']]

            else:
                new_actions[i] = actions[i]

            i = i + 1

        return new_actions
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            #print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            #print("total_cost: ", self.cost)
            #print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            #print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('results/account_rewards_train.csv')
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            
            # modifying actions according to the sentiment
            # print("#### " + str(self.trades))
            # print("before")
            # print(actions)
            # actions = self.modifyActions(actions)
            # actions = np.array(actions)
            # print("after: ")
            # print(actions)

            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING



        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() 
        # iteration += 1 
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
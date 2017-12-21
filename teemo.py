import pandas as pd
import requests
import json
from sklearn.externals import joblib
import pickle

class GetGameData(object):
    '''
    This is a class for getting gameplay data from the API by giving it
    a key (expires every 24 hrs), and then using the generate_x object to
    generate data from the API.

    Note: Riot API has very strict data limiting, (per second, per minute, per day)
    .generate_champ_mastery(list of names)
    .generate_league(list of names)
    .generate_position(list of names)
    .generate_masteries(list of names)
    .generate_match_data(list of names)
    .predict_my_wins(list of names)

    '''

    def __init__(self, key):
        self.key = key

    def generate_id(self, names): # list
        ids = {}
        for i in names:
            temp_dict = {}
            api = '/lol/summoner/v3/summoners/by-name/{}'.format(i)
            content = json.loads(requests\
                    .get('https://na1.api.riotgames.com/{}?api_key={}'\
                    .format(api, self.key)).content)
            temp_dict['accountId'] = content['accountId']
            temp_dict['summonerId'] = content['id']
            ids[i] = temp_dict
        return ids

    def generate_champ_mastery(self, names):
        id_dict = self.generate_id(names) # dict
        placeholder = []
        for k, v in id_dict.items():
            api = '/lol/champion-mastery/v3/champion-masteries/by-summoner/{}'\
                                                        .format(v['summonerId'])
            content = json.loads(requests\
                .get('https://na1.api.riotgames.com/{}?api_key={}'\
                .format(api, self.key)).content)
            placeholder.append(content)
        dataframe = pd.concat([pd.DataFrame(placeholder[i]) for i in range(len(placeholder))])
        return dataframe

    def generate_league(self, names):
        id_dict = self.generate_id(names) # {name : {accountId : value, summonerId : value}}
        placeholder = []
        for k, v in id_dict.items():
            api = '/lol/league/v3/leagues/by-summoner/{}'\
                                        .format(v['summonerId'])
            content = json.loads(requests\
                .get('https://na1.api.riotgames.com/{}?api_key={}'\
                .format(api, self.key)).content)
            placeholder.append(content)
        dataframe = pd.concat([pd.DataFrame(placeholder[i]) for i in range(len(placeholder))])
        return dataframe

    def generate_position(self, names):
        id_dict = self.generate_id(names) # dict
        placeholder = []
        for k, v in id_dict.items():
            api = '/lol/league/v3/positions/by-summoner/{}'\
                                        .format(v['summonerId'])
            content = json.loads(requests\
                .get('https://na1.api.riotgames.com/{}?api_key={}'\
                .format(api, self.key)).content)
            placeholder.append(content)
        dataframe = pd.concat([pd.DataFrame(placeholder[i]) for i in range(len(placeholder))])
        return dataframe

    def generate_masteries(self, names):
        id_dict = self.generate_id(names) # dict
        placeholder = []
        for k, v in id_dict.items():
            api = '/lol/platform/v3/masteries/by-summoner/{}'\
                                        .format(v['summonerId'])
            content = json.loads(requests\
                .get('https://na1.api.riotgames.com/{}?api_key={}'\
                .format(api, self.key)).content)
            placeholder.append(content)
        dataframe = pd.concat([pd.DataFrame(placeholder[i]) for i in range(len(placeholder))])
        return dataframe

    def generate_match_data(self, names):
        id_dict = self.generate_id(names) # dict
        placeholder = []
        for k, v in id_dict.items():
            api = '/lol/match/v3/matchlists/by-account/{}'.format(v['accountId'])
            content = json.loads(requests\
                .get('https://na1.api.riotgames.com/{}?api_key={}'\
                .format(api, self.key)).content)
            placeholder.append(content)
        dataframe = pd.concat([pd.DataFrame(placeholder[i]) for i in range(len(placeholder))])
        return dataframe

    def generate_player_history(self, names):
        games = self.generate_match_data(names)
        placeholder = []
        game_id = []
        for i in [x for x in games['matches'].values]:
            game_id.append(i['gameId'])
        for i in game_id:
            api = '/lol/match/v3/matches/{}'.format(i)
            content = json.loads(requests.get('https://na1.api.riotgames.com/{}?api_key={}'\
                    .format(api, self.key)).content)
            placeholder.append(content)
        history_list = []
        for i in placeholder:
            for x in range(0, 9): # participant id's (1 - 10) needed to access player stats
                try:
                    if i['participantIdentities'][x]['player']['summonerName'] in names:
                        history_list.append(i['participants'][x]['stats'])
                except:
                    break # iteration ends without KeyError
        return pd.DataFrame(history_list)

    def predict_my_wins(self, names):
    	'''
    	Fetches user data from Riot API server, connects it with trained model to make predictions
    	Returns prediction numpy array

    	'''
        print('Fetching data...')
        hist = self.generate_player_history(names) # this is a dataframe of player history

        col_list = ['win', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'kills',
        'deaths', 'assists', 'largestkillingspree', 'largestmultikill',
        'killingsprees', 'longesttimespentliving', 'doublekills', 'triplekills',
        'quadrakills', 'pentakills', 'magicdmgdealt', 'largestcrit',
        'magicdmgtochamp', 'dmgselfmit', 'dmgtoobj', 'dmgtoturrets',
        'visionscore', 'magicdmgtaken', 'goldearned', 'goldspent',
        'turretkills', 'inhibkills', 'neutralminionskilled', 'ownjunglekills',
        'enemyjunglekills', 'champlvl', 'wardsplaced', 'wardskilled',
        'firstblood']

        change_dict = {
        'champlevel' : 'champlvl',
        'damagedealttoturrets':'dmgtoturrets',
        'damageselfmitigated' : 'dmgselfmit',
        'firstbloodkill' : 'firstblood',
        'inhibitorkills' : 'inhibkills',
        'largestcriticalstrike' : 'largestcrit',
        'magicdamagedealt' : 'magicdmgdealt',
        'magicdamagedealttochampions' : 'magicdmgtochamp',
        'magicaldamagetaken' : 'magicdmgtaken',
        'neutralminionskilledenemyjungle' : 'enemyjunglekills',
        'neutralminionskilledteamjungle' : 'ownjunglekills',
        'damagedealttoobjectives' : 'dmgtoobj'}
        hist.columns = hist.columns.map(lambda x: x.lower())
        hist = hist.fillna(0)
        hist = hist.rename(columns=change_dict)

        print('Reshaping Data...')

        for col in hist.columns:
            if col not in col_list:
                hist.drop(col, axis = 1, inplace = True)

        # reorder one of the DataFrames to match the order of columns
        feed = pd.DataFrame()
        for col in hist.columns:
            feed[col] = hist[col]

        print('Making predictions....')

        clf = pickle.load(open('model.sav', 'rb'))
        
        preds = clf.predict(feed.drop('win', axis=1))
        pred_proba = clf.predict_proba(feed.drop('win', axis=1))
        r2_score = clf.score(feed.drop('win', axis=1), feed.win)
        print('predictions = ', preds)
        print('predict_probas = ', pred_proba)
        print('r2 score = ', r2_score)

        return preds

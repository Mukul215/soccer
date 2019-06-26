import sys
import os
import datetime as dt
import pandas as pd
import statsmodels.api as sm  # needed for Poisson regression model
import statsmodels.formula.api as smf  # needed for Poisson regression model
import numpy as np
from scipy.stats import poisson, skellam

# Convert moneyline and adds from imlied value to fair value


def oddsConverter(probability):
    return round(1/probability, 2)


def moneylineConverter(probability):
    if probability >= .5:
        return round(-((probability/(1-probability))*100), 0)
    else:
        return round((((1-probability)/probability)*100), 0)


df = pd.read_csv(
    "http://www.football-data.co.uk/mmz4281/1819/I2.csv", encoding='ISO-8859-1')

# Change teams here
homeTeam = "Salernitana"
awayTeam = "Venezia"

df['Date'] = pd.to_datetime(df['Date'])
df_include = df[df['Date'].dt.year >= 2018]
df_calculate = df_include[['Date', 'HomeTeam',
                           'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

goal_model_data = pd.concat([df_calculate[['HomeTeam', 'AwayTeam', 'FTHG']].assign(home=1).rename(
    columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'}),
    df_calculate[['AwayTeam', 'HomeTeam', 'FTAG']].assign(home=0).rename(
    columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'})])

# calculate poisson model
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()
home_team = poisson_model.predict(pd.DataFrame(data={'team': homeTeam, 'opponent': awayTeam,
                                                     'home': 1}, index=[1]))
away_team = poisson_model.predict(pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam,
                                                     'home': 0}, index=[1]))

# create a matrix of 5x5 (goals) and probabilities
max_goals = 5
team_pred = [[poisson.pmf(i, team_avg) for i in range(
    0, max_goals+1)] for team_avg in [home_team, away_team]]

model_array = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

# calculate the probability that home team wins
homeWin = model_array[1][0] + model_array[2][0] + model_array[2][1] + model_array[3][0] + model_array[3][1] + model_array[3][2] + \
    model_array[4][0] + model_array[4][1] + model_array[4][2] + model_array[4][3] + model_array[5][0] + model_array[5][1] + model_array[5][2] + \
    model_array[5][3] + model_array[5][4]

# calculate the probability that there is a draw
outcomeDraw = model_array[0][0] + model_array[1][1] + model_array[2][2] + \
    model_array[3][3] + model_array[4][4] + model_array[5][5]

# calculate the probability that the away team wins
awayWin = model_array[0][1] + model_array[0][2] + model_array[0][3] + model_array[0][4] + model_array[0][5] + model_array[1][2] + model_array[1][3] + \
    model_array[1][4] + model_array[1][5] + model_array[2][3] + model_array[2][4] + model_array[2][5] + model_array[3][4] + model_array[3][5] + \
    model_array[4][5]

# exact goals probability
exact2 = model_array[0][2] + model_array[1][1] + model_array[2][0]
exact3 = model_array[0][4] + model_array[4][0] + \
    model_array[3][1] + model_array[1][3]

# calculate the probability that the match outcome is over/under a certain amount
under25 = model_array[0][0] + model_array[0][1] + model_array[1][0] + model_array[1][1] + \
    model_array[2][0] + model_array[0][2]
over25 = 1 - under25
under35 = under25 + \
    model_array[0][3] + model_array[3][0] + \
    model_array[2][1] + model_array[1][2]
over35 = 1 - under35

# Convert all probabilities into moneyline and odds
moneylineExact2 = moneylineConverter(exact2)
decimalExact2 = oddsConverter(exact2)
moneylineDraw = moneylineConverter(outcomeDraw)
decimalDraw = oddsConverter(outcomeDraw)
moneylineHome = moneylineConverter(homeWin)
decimalHome = oddsConverter(homeWin)
moneylineAway = moneylineConverter(awayWin)
decimalAway = oddsConverter(awayWin)
moneylineUnder25 = moneylineConverter(under25)
decimalUnder25 = oddsConverter(under25)
moneylineOver25 = moneylineConverter(over25)
decimalOver25 = oddsConverter(over25)
moneylineExact3 = moneylineConverter(exact3)
decimalExact3 = oddsConverter(exact3)
moneylineOver35 = moneylineConverter(over35)
decimalOver35 = oddsConverter(over35)
moneylineUnder35 = moneylineConverter(under35)
decimalUnder35 = oddsConverter(under35)

# print(poisson_model.summary())
print("Home team: {}".format(home_team))
print("Away team: {}\n\n".format(away_team))
# print(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
print("Implied Probability {} Wins: {:.2%}".format(homeTeam, homeWin))
print("Implied Probability of Draw: {:.2%}".format(outcomeDraw))
print("Implied Probability {} Wins: {:.2%}".format(awayTeam, awayWin))
print(
    "Implied Probability Game Total is Under 2.5: {:.2%}".format(under25))
print(
    "Implied Probability Game Total is Exactly 2: {:.2%}".format(exact2))
print("Implied Probability Game Total is Over 2.5: {:.2%}".format(over25))
print(
    "Implied Probability Game Total is Under 3.5: {:.2%}".format(under35))
print("Implied Probability Game Total is Exactly 3: {:.2%}".format(exact3))
print("Implied Probability Game Total is Over 3.5: {:.2%}".format(over35))
print("\n\nFair Value of {} Wins: {} and odds at: {}".format(homeTeam,
                                                             moneylineHome, decimalHome))
print("Fair Value of Draw: {} and odds at: {}".format(
    moneylineDraw, decimalDraw))
print("Fair Value of {} Wins: {} and odds at: {}".format(awayTeam,
                                                         moneylineAway, decimalAway))
print("Fair Value of Game Total Under 2.5: {} and odds at: {}".format(
    moneylineUnder25, decimalUnder25))
print("Fair Value of Game Total Exactly 2: {} and odds at: {}".format(
    moneylineExact2, decimalExact2))
print("Fair Value of Game Total Over 2.5: {} and odds at: {}".format(
    moneylineOver25, decimalOver25))
print("Fair Value of Game Total Under 3.5: {} and odds at: {}".format(
    moneylineUnder35, decimalUnder35))
print("Fair Value of Game Total Exactly 3: {} and odds at: {}".format(
    moneylineExact3, decimalExact3))
print("Fair Value of Game Total Over 3.5: {} and odds at: {}".format(
    moneylineOver35, decimalOver35))
print("\n\nPlease check betfair exchanges for the most accurate lines\n\n\n")

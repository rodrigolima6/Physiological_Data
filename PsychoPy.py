import pandas as pd
from csv_load import *

def Horror(data):
    horror = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Horror' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'pereived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        horror[users] = x

        for users in horror.keys():
            horror[users] = horror[users].sort_values('videos')

    return horror

def Erotic(data):
    erotic = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Erotic' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'pereived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        erotic[users] = x

        for users in erotic.keys():
            erotic[users] = erotic[users].sort_values('videos')

    return erotic

def Scenery(data):
    scenery = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Scenery' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'pereived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        scenery[users] = x

        for users in scenery.keys():
            scenery[users] = scenery[users].sort_values('videos')

    return scenery

def Social_positive(data):
    social_positive = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Social Positive' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'pereived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        social_positive[users] = x

        for users in social_positive.keys():
            social_positive[users] = social_positive[users].sort_values('videos')

    return social_positive

def Social_negative(data):
    social_negative = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Social Negative' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'pereived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        social_negative[users] = x

        for users in social_negative.keys():
            social_negative[users] = social_negative[users].sort_values('videos')

    return social_negative

# path = 'G:\\O meu disco\\PhD\\1st Study\\PsychoPy Data\\'
# data = CSV_Load(path)
# horror = Horror(data)


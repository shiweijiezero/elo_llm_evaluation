import numpy as np
from random import choices
from itertools import combinations
import math
import utils

# 1. 首先初始化所有选手，根据历史对战记录初始化选手得分，获得选手的答题记录
initial_elo = 1500  # ELO初始分，如果新选手不在历史选手中，则使用初始化得分
K = 32  # 浮动系数，每场比赛能得到的最大分数，这个值应该视选手的资力而定。资力越深，参加的比赛应该越多，因此K值应该越低。通常业余选手为30，专业为15，大师为10，特级大师为5。
elo_dict = utils.get_elo_scores("battle_records",initial_elo,K) # {"player_name":[elo_score,win_count,all_count]}

# # 2. 选择进行新一轮比赛的选手范围，以及选手主场作战的出场次数
# candidate_players = ["Qwen"]
# player_round = [3]

# 2. 随机选择实力相近的选手进行比赛，并更新选手ELO评分，更新battle记录


# 3. 可视化battle与ELO评分变化过程

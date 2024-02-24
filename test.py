import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choices
from itertools import combinations
import math

plt.style.use('dark_background')

# Initial parameters
n_players = 5
initial_elo = 1500
n_rounds = 2000
K = 1

# Define window size for running average
window_size = 50

# Set up bias factors
bias_factors = [0.6, 0.8, 1.0, 1.2, 1.4]


# Function to calculate expected score
def expected_score(rating1, rating2):
    return 1 / (1 + 10**((rating2 - rating1) / 400))

# Function to update ratings after a game
def update_elo(rating1, rating2, score1, K):
    expected1 = expected_score(rating1, rating2)
    return rating1 + K * (score1 - expected1)

def compute_expected_elo(rating, prob_of_winning):
    return rating + 400 * math.log10(prob_of_winning)
    # return rating + 400 * np.log10(1 / prob_of_winning - 1)


expected_elo = [compute_expected_elo(initial_elo, bias_factor) for bias_factor in bias_factors]

# Reset initial ratings
ratings = [initial_elo for _ in range(n_players)]

# Simulate rounds with bias
history = [list(ratings)]
# for i in range(n_players):
#     for j in range(i + 1, n_players):
for _ in range(n_rounds):
    for i, j in combinations(range(n_players), 2):
        # Determine game outcome based on current ratings and bias factors
        players = [i, j]
        # weights = [ratings[i] * bias_factors[i], ratings[j] * bias_factors[j]]
        weights = [bias_factors[i], bias_factors[j]]
        winner = choices(players, weights=weights, k=1)[0]

        # Update ratings
        if winner == i:
            ratings[i] = update_elo(ratings[i], ratings[j], 1, K)
            ratings[j] = update_elo(ratings[j], ratings[i], 0, K)
        else:
            ratings[i] = update_elo(ratings[i], ratings[j], 0, K)
            ratings[j] = update_elo(ratings[j], ratings[i], 1, K)

    # Record ratings after this round
    history.append(list(ratings))

# Convert history to DataFrame for easy plotting
history_df = pd.DataFrame(history, columns=[f'Player {i + 1}' for i in range(n_players)])

# Create a colormap
colors = plt.get_cmap('Set3')  # tab10

# Apply running average filter
history_df_smooth = history_df.rolling(window_size).mean()

# Plotting the ELO ratings over time with bias factors and more rounds
plt.figure(figsize=(10, 6))
for i, player in enumerate(history_df_smooth.columns):
    # Plot smoothed data
    plt.plot(history_df_smooth[player], label=player + ' (Smoothed)', color=colors(i))
    # Plot original data with lower opacity
    plt.plot(history_df[player], alpha=0.3, color=colors(i))
    # Plot expected Elo rating
    plt.axhline(y=expected_elo[i], linestyle='--', color=colors(i))  # , label=player + ' (Expected)'

plt.xlabel('Round')
plt.ylabel('Elo Rating')
plt.title(f'Simulated Elo Ratings Over Time ({n_rounds} Rounds, Smoothed)')
plt.legend()
# plt.grid(True)
plt.show()
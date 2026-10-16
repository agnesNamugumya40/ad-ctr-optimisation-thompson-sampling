# Thompson Sampling — Ad CTR Optimisation

A reinforcement learning project that uses the **Thompson Sampling** algorithm to find which of 10 ad versions has the highest click-through rate (CTR), learning from user responses one round at a time.

## What is Thompson Sampling?

Thompson Sampling solves the **multi-armed bandit problem**: there are several options (here, 10 ads), each with an unknown reward rate, and the goal is to collect as many rewards (clicks) as possible while still learning which option is best.

Instead of keeping a single estimate for each ad, Thompson Sampling keeps a **probability distribution** of what each ad's true CTR could be. For each ad, it tracks:

- **numbers_of_rewards_1** — how many times the ad was clicked
- **numbers_of_rewards_0** — how many times the ad was shown but not clicked

In each round, it draws a random value for every ad from a **Beta distribution**:

```
random_beta = betavariate(numbers_of_rewards_1 + 1, numbers_of_rewards_0 + 1)
```

and shows the ad with the highest draw. Ads with little data have wide distributions, so they sometimes produce high draws and get explored. Ads with lots of data have narrow distributions centred on their real CTR, so the best ad wins most rounds.

## Dataset

**File:** `Ads_CTR_Optimisation.csv`

- 10,000 rows, each representing one user (one round)
- 10 columns (`Ad 1` to `Ad 10`), one per ad version
- Each value is `1` if the user would click that ad and `0` if not

The dataset is a simulation: in real life we only learn the response for the ad that was actually shown, and the algorithm only ever reads that one value per round.

## Project Structure

```
├── thompson_sampling.ipynb        # Main notebook
├── Ads_CTR_Optimisation.csv       # Simulated user responses to 10 ads
└── README.md                      # This file
```

## Workflow

1. **Import libraries** — NumPy, Matplotlib, Pandas
2. **Load dataset** — reads `Ads_CTR_Optimisation.csv`
3. **Implement Thompson Sampling** — for each of the 10,000 rounds, draws a random value from each ad's Beta distribution, selects the ad with the highest draw, and updates that ad's click / no-click counts
4. **Visualise results** — histogram showing how many times each ad was selected

## Dependencies

```bash
pip install numpy pandas matplotlib
```

## How to Run

1. Open `thompson_sampling.ipynb` in Jupyter Notebook or Google Colab.
2. Make sure `Ads_CTR_Optimisation.csv` is in the same folder (or uploaded to Colab).
3. Run all cells in order.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 10000 | Number of rounds (users) |
| `d` | 10 | Number of ads |
| Prior | Beta(1, 1) | Every ad starts with a uniform prior (no assumptions) |

## Results

Because Thompson Sampling is random, the exact numbers change slightly on each run. Across 20 runs:

- **Ad 5** was identified as the best ad **every time**, selected in roughly **89–95%** of rounds (about 8,900–9,500 of the 10,000).
- **Total reward: about 2,580–2,630 clicks** per run, compared with roughly 1,200 clicks from showing ads at random.
- The algorithm usually locks onto Ad 5 within the **first 500 rounds**.

Example run:

| Ad | Times selected |
|----|---------------:|
| Ad 1 | 77 |
| Ad 2 | 60 |
| Ad 3 | 54 |
| Ad 4 | 37 |
| **Ad 5** | **9,413** |
| Ad 6 | 26 |
| Ad 7 | 72 |
| Ad 8 | 204 |
| Ad 9 | 25 |
| Ad 10 | 32 |

Total reward in this run: **2,620 clicks**.

## Thompson Sampling vs. UCB

On the same dataset, Thompson Sampling outperforms Upper Confidence Bound:

| | UCB | Thompson Sampling |
|---|---|---|
| Best ad found | Ad 5 | Ad 5 |
| Rounds needed to identify it | ~1,000 | ~500 |
| Total reward (10,000 rounds) | 2,178 | ~2,600 |
| Type | Deterministic | Probabilistic |

## Key Concepts

- **Exploration vs. exploitation**: trying under-tested ads to learn about them versus showing the ad that currently looks best.
- **Beta distribution**: a probability distribution over values between 0 and 1, used here to represent the uncertainty about each ad's true CTR.
- **Bayesian updating**: each click or non-click updates the ad's distribution, making the estimate sharper over time.
- **Probabilistic**: the algorithm uses random draws, so results vary slightly between runs.

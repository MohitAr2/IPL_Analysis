# 🏏 IPL Analysis — EDA, Candlestick Performance Charts & 2025 Points Table Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-4C72B0?style=flat)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> Full end-to-end analysis of **1,095 IPL matches** and **260,000+ ball-by-ball deliveries** from **2008 to 2024** — covering exploratory data analysis, a novel candlestick chart adaptation for cricket performance, and a weighted multi-feature linear regression model to predict the **2025 IPL Points Table**.

---


## 📦 Dataset

**Source:** [IPL Complete Dataset 2008–2020 — Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

| File | Rows | Description |
|------|------|-------------|
| `matches.csv` | 1,095 | Match-level metadata — venue, toss, winner, result margin, target |
| `deliveries.csv` | ~260,000 | Every delivery — batter, bowler, runs, extras, over |

**Matches columns used:**
`id · season · city · date · venue · team1 · team2 · toss_winner · toss_decision · winner · result · result_margin · target_runs · target_overs · super_over · player_of_match`

**Deliveries columns used:**
`match_id · inning · batting_team · bowling_team · over · ball · batter · bowler · batsman_runs · extra_runs · total_runs`

---

## 🧹 Data Cleaning & Preprocessing

- Dropped redundant columns: `method`, `umpire1`, `umpire2`
- Normalised franchise name aliases across seasons:
  - `Delhi Daredevils` → `Delhi Capitals`
  - `Kings XI Punjab` → `Punjab Kings`
  - `Rising Pune Supergiant` → `Rising Pune Supergiants`
- Parsed mixed date formats (`DD/MM/YYYY` and `YYYY-MM-DD`)
- Handled all three result types in the points system:

| Result Type | Logic | Points |
|-------------|-------|--------|
| Normal win | `winner` col present, no super-over | Winner **+2**, Loser **+0** |
| Super-over (tie) | `winner` col has super-over winner | Winner **+2**, Loser **+0** |
| No result | `result == 'No result'`, `winner` is NaN | Both teams **+1** |

---

## 📊 Notebook 1 — NumPy Structured Analysis

> **Rules:** No Pandas. NumPy + core Python only. Vectorised operations throughout.

Key technique used across all tasks:

```python
# Broadcast mask matrix replaces all groupby loops
mask_matrix = entity_array == unique_entities[:, None]   # (N_entities × N_deliveries)
totals      = mask_matrix @ values_array                 # dot product = grouped sum
```

| Task | Analysis |
|------|----------|
| 0 | `np.genfromtxt` load with `dtype=None`, `filling_values` |
| 1 | Total runs per match via broadcast + dot product |
| 2 | Top 5 batters — `np.unique` + `np.argsort` |
| 3 | Strike rate — balls faced via `.sum(axis=1)`, `np.where` divide guard |
| 4 | Economy rate — `balls / 6.0` overs conversion |
| 5 | Avg runs per over — per-over broadcast |
| 6 | Boundary analysis — `batsman_runs == 4/6` boolean masks |
| 7 | Death overs — `(over >= 16) & (over <= 20)` combined condition |
| 8 | Highest scoring match — `np.argmax` |
| 9 | Match winner — `np.char.add` composite key + `np.bincount(weights=...)` |
| 10 | Toss impact — `np.searchsorted` manual join across datasets |
| 11 | Scorecard generation — grouped aggregation without dicts |

---

## 🔬 Notebook 2 — Pandas EDA Pipeline

Full data pipeline: **Ingest → Clean → Transform → Analyse → Report → Export**

### Derived Columns

```python
df['is_four']       = (df['batsman_runs'] == 4).astype(int)
df['is_six']        = (df['batsman_runs'] == 6).astype(int)
df['is_dot']        = (df['batsman_runs'] == 0).astype(int)
df['is_powerplay']  = df['over'].between(1, 6).astype(int)
df['is_death_over'] = df['over'].between(16, 20).astype(int)
```

### Analyses Covered

| # | Analysis | Key Insight |
|---|----------|-------------|
| 1 | Total runs per match | Distribution + top 15 highest scoring matches |
| 2 | Runs per team per match | Avg, max, std — boxplot by team |
| 3 | Top 10 batters | Overall run tally |
| 4 | Strike rate | Min 200 balls qualifier — scatter vs total runs |
| 5 | Economy rate | Min 20 overs qualifier — most economical bowlers |
| 6 | Most consistent batters | Consistency score = avg / (std + 1) |
| 7 | Highest individual match score | Top 20 innings |
| 8 | Boundary analysis | Fours vs sixes split, top boundary hitters |
| 9 | Boundary percentage | % of runs from boundaries per batter |
| 10 | Dot ball analysis | Bowlers with most dots + dot ball % |
| 11 | Runs per over (1–20) | Powerplay vs death over run rate trend |
| 12 | Powerplay performance | Best teams in overs 1–6 |
| 13 | Death overs performance | Best teams and batters in overs 16–20 |
| 14 | 1st vs 2nd innings | Violin plot comparison |
| 15 | Toss impact | Bat first vs field first win rates |
| 16 | Season-wise trends | Total runs and avg run rate per season |
| 17 | Winner approximation | By runs scored per innings |
| 18 | Correlation heatmap | Feature relationships across all derived cols |

---

## 🕯️ Notebook 3 — Candlestick Charts + Points Table + 2025 Prediction

### Year-wise Points Table

Built a full IPL-style points table for every season from scratch using match metadata and deliveries:

```
Points  : Win = 2  |  No Result = 1 each  |  Loss = 0
NRR     : (runs_scored / balls_batted − runs_conceded / balls_bowled) × 6
```

### 🕯️ Candlestick Chart — Cricket Adapted from Finance

One of the more unconventional choices in this project. Candlestick charts are typically used to analyse **stock market price movements** for a company or ETF over time — the body of a candle represents the open/close price and the wicks show the high/low for that period.

Here, the same structure is mapped to **IPL team batting performance** across a season:

| Financial Concept | Cricket Mapping |
|---|---|
| **Open** | Team's runs in their **first game** of the season |
| **Close** | Team's runs in their **last game** of the season |
| **High (wick)** | **Best** batting total in any single match that season |
| **Low (wick)** | **Worst** batting total in any single match that season |
| 🟩 **Green body** | Team scored more in their last game than their first — form improving |
| 🟥 **Red body** | Team scored less toward the end — form dropping |

Two chart views are provided:
- **Per-franchise across all seasons** — 2×4 grid showing each team's full history
- **All teams in a single season** — side-by-side comparison for any given year

This makes it visually obvious which teams hit their peak early (false start) vs teams that peaked in the playoffs phase — something a standard bar chart doesn't capture.

### 📐 Weighted Multi-Linear Regression — 2025 Prediction

Features are **StandardScaled** then multiplied by domain weights before the model sees them, which forces explicit importance ordering without relying on the model to infer it from correlated data alone.

```python
X_weighted = StandardScaler().fit_transform(X_raw) * DOMAIN_WEIGHTS
model      = LinearRegression().fit(X_weighted, y_points)
```

| Feature | Domain Weight | Rationale |
|---------|:---:|-----------|
| `avg_runs_scored` (batting run rate) | **2.5** | Batting firepower is the strongest single predictor of wins |
| `avg_runs_conceded` (bowling economy) | **2.0** | Restricting the opposition is equally match-defining |
| `avg_nrr` | **1.8** | NRR is a composite of both — directly tied to standings |
| `win_pct` | **1.5** | Recent form momentum carries into prediction |
| `avg_result_margin` | **0.9** | Dominant wins signal depth, not just close calls |
| `toss_win_pct` | **0.6** | Marginal advantage — home/away conditions matter more |

**2025 input features** are computed as the 3-season rolling average (2018–2020) per franchise. New franchises without historical data (Gujarat Titans, Lucknow Super Giants) are seeded at dataset median values with a small positive bias.

### Model Performance

| Metric | Value |
|--------|-------|
| R² (training) | reported in notebook |
| R² (5-fold CV) | reported in notebook |
| MAE | reported in notebook (points) |

---

## 📈 Visual Gallery

All plots use a consistent **dark IPL theme** with franchise-accurate hex color codes for all 16 teams.

| Visual | Description |
|--------|-------------|
| Points heatmap | Team × Season grid — instant cross-season comparison |
| Win % trajectory | Line chart per franchise across all seasons |
| Season ranking heatmap | Green = table topper, Red = bottom |
| Candlestick grid | 2×4 — one candle per season per franchise |
| Candlestick season view | All teams in one season on a single axis |
| Predicted points bar | 2025 table with predicted wins |
| Batting vs bowling bubble | Runs scored vs conceded — bubble = predicted points |
| Historical + 2025 trend | Dashed projection lines from last known season to 2025 |
| Radar charts | Top 6 predicted teams — multi-dimensional profile |
| NRR vs points scatter | With regression line — all historical seasons |
| Toss impact analysis | Win % when toss won + bat/field decision trend |
| Venue scoring chart | Top 15 venues by avg match runs |

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/yourusername/ipl-analysis.git
cd ipl-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter

# Download the dataset from Kaggle and place CSVs in data/

# Run notebooks in order
jupyter notebook
```

**Python version:** 3.10+  
**No GPU required** — all computation is CPU-bound NumPy/Pandas/sklearn.

---

## 📤 Outputs

Running all three notebooks generates:

```
output/
├── runs_per_match.csv               # Match-level run totals
├── top_batters.csv                  # Top 10 batters by total runs
├── strike_rate.csv                  # All qualified batters with SR
├── economy.csv                      # All qualified bowlers with economy
├── team_scores.csv                  # Team avg/max/total per season
├── death_overs.csv                  # Top death-over batters
├── yearwise_points_long.csv         # Long-form points table all seasons
├── feature_table_all_seasons.csv    # Feature matrix used for MLR
├── predicted_2025_points_table.csv  # Final 2025 prediction
└── ipl_points_analysis.xlsx         # All season tables + prediction (multi-sheet)
```

---

## 🧠 Key Findings

- Toss-to-win conversion hovers around **50%** across most seasons — toss is far less predictive than batting or bowling quality
- Death over (16–20) run rates have increased consistently season-on-season, rewarding teams with explosive finishers
- NRR and points have a strong positive linear relationship — teams with NRR > +0.5 almost always make playoffs
- The candlestick view reveals that most title-winning teams show **green candles in the second half** of the season — they peak at the right time
- Bowling economy (runs conceded rate) is nearly as predictive as batting run rate for final standings

---

## 📄 License

MIT License — free to use, adapt, and build on with attribution.

---

## 🙏 Acknowledgements

- Dataset by [Patrick B](https://www.kaggle.com/patrickb1912) on Kaggle
- IPL match data sourced from ESPNcricinfo
- Candlestick visualisation concept adapted from financial charting (OHLC)

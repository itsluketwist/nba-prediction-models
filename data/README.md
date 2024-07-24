# data

This module contains NBA game result sequential datasets, created from a database of NBA game 
results and statistics. All data creation and manipulation code is included.

A description of the data format, along with choices made regarding the selection of which data 
to use are included below.

## contents

- `create.ipynb` - Notebook containing the code used to create the datasets from the raw data.
- `raw_csv/*` - Contains csv files of the main data used (courtest of [Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball/data)).
- `parquet/*` - Contains `.parquet` files of dataframes with the testing / training / evaluation datasets.
  - `complete_df.parquet` - All available game data.
  - `training_df.parquet` - Training dataset (games before the 2022 All-star break).
  - `evaluation_df_all.parquet` - Evaluation dataset (games after the 2022 All-star break).
  - `evaluation_df_half_21_22.parquet` - The 2021-22 season games from the evaluation data.
  - `evaluation_df_full_22_23.parquet` - The 2022-23 season games from the evaluation data.
  - `final_week_df.parquet` - Last week of data from the 2022-23 season, as small sample/example data.
  - `training_streaks_short_df.parquet` - Dataset of streak-breaks of length 3-4, from training data.
  - `training_streaks_long_df.parquet` - Dataset of streak-breaks of length 5+, from training data.
  - `evaluation_streaks_short_df.parquet` - Dataset of streak-breaks of length 3-4, from evaluation data.
  - `evaluation_streaks_long_df.parquet` - Dataset of streak-breaks of length 5+, from evaluation data.


## source and license

The raw data used is from the comprehensive 
[NBA Database](https://www.kaggle.com/datasets/wyattowalsh/basketball/data) dataset found on Kaggle.

Some additional metrics are calculated to supplement the data, and it was molded into sequential 
datasets for model training purposes. None of the data itself has been changed.

The new datasets are available to use under the 
[Creative Commons BY SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).

## dataset format

The datasets have records that correspond to a specific NBA game, contianing the result and 
statistics from the 10 previous games for both teams.

Each dataset is stored as a pandas dataframe in a parquet file, and can be loaded with:

```
import pandas as pd

df = pandas.load_parquet("path/to/file.parquet")
```

The dataframe has 2 columns:

- `"info"`: A dictionary of human-readable data about the game.
- `"data"`: A 10 x 116 tensor, representing features from the previous 10 games by each team, ordered chronologically.

If the record is for the game HomeTeam vs AwayTeam, each row of the `data` tensor has the following structure, by index:
- `0-3`: General game information for HomeTeam's previous game.
- `4-30`: HomeTeam statistics for that game.
- `31-57`: HomeTeam's opponent's statistics for that game.
- `58-61`: General game information for AwayTeam's previous game.
- `62-88`: AwayTeam statistics for that game.
- `89-115`: AwayTeam's opponent's statistics for that game.

A full breakdown of the exact features included in the `data` tensor is available in the 'which data?' section below.


## which seasons?

There is widely regarded to be [three generations](https://content.iospress.com/articles/journal-of-sports-analytics/jsa200525) of the NBA regarding gameplay, since the introduction of the 3-point line in [1979](https://www.nba.com/news/this-day-in-history-oct-12-the-first-3-point-field-goal):
- classic 1979-1994
- transitional 1995-2013
- modern 2014 onwards

Modern gameplay is more emphasized on condensing the space on the court from which shots are attempted, with a big move to either 3-point shooting or layups, and less mid-range.
Using only data from the modern era will be limiting (only 8 seasons available), and this gameplay shift was really triggered by the [introduction of Steph Curry](https://core.ac.uk/download/pdf/145239674.pdf) to the NBA - so we'll use data from the 2009-10 season onwards.

First game of 2009-10 season happened on: `27 Oct 2009`


## sequence length?

It's important to provide the correct amount of sequential games in order for the models to learn.

- 3 or more wins in a row is considered to be a streak according to [Wikipedia](https://en.wikipedia.org/wiki/Winning_streak)
- [Analysis of winning streaks](https://www.scirp.org/journal/paperinformation?paperid=74910) in the 2016 regular season gives a high average winning streak of 7.3, and a high average losing streak of 6.55.

I'll construct datasets that are 10 games in length, and train models on 5-game and 10-game sequences.

I'll consider streaks to be 3 or more wins/losses in a row, and also create datasets of streak-breaks for streaks
of length 3-4, and of length 5+.


## training and evaluation split?

In order to determine if the model generalises well, and will be able to predict game results in the future - I'll hold back the most modern data.

- The whole latest season of data (for 2022-23) will be used for evaluation - to check generalisation to a new season.
- The second half of the 2021-22 season will be used for evaluation - to check generalisation within a season.
- Data after the 2022 All-star break, on 21 Feb 2022, will be used for evaluation.
- Data from the final week of the 2023 season, starting 4 Apr 2023, will be used as an example sample.


## which data?

The following data will be used to train the models, full descriptions of each statistic can be found in the [NBA glossary](https://www.nba.com/stats/help/glossary).

- Game stats (4)
  - Game result
  - Play at home
  - Close game
  - Overtime count

- Per-Team stats (27)
  - Season (2)
    - Games played
    - Win %
  - Line score (6)
    - 1st quarter points
    - 2nd quarter points
    - 3rd quarter points
    - 4th quarter points
    - Overtime points
    - Total points
  - Play stats (19)
    - Field-goals made
    - Field goals %
    - 3-point made
    - 3-point %
    - Free-throws made
    - Free-throws %
    - Total rebounds
    - Offensive rebounds
    - Defensive rebounds
    - Assists
    - Steals
    - Blocks
    - Turnovers
    - Personal fouls
    - Plus-minus
    - Points in the paint
    - Second chance points
    - Fast break points
    - Largest lead

Notes:
 - Data has index 0 = oldest game, index 9 = closest to game to be predicted.
 - Each piece of sequential data will contain the above metrics for both the home and away team in question. Leading to feature vectors of size 116.

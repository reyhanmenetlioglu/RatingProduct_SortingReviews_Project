# Calculating Average Rating According to Current Comments
# & Comparing with Existing Average Rating

# region import & read

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('Measurement_Problems/Rating_Products_Sorting_Revies_With_Amazon/dataset/amazon_review.csv')

# endregion

# region Average val. of the product

df['overall'].mean()

# endregion

# region Converting type of df['reviewTime'] to datetime64[ns]

df.head()
df.info()

df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df.info()

# endregion

# region Creating current_day, df['days']

df['reviewTime'].max()
current_date = pd.to_datetime('2014-12-07 00:00:00')

df['days'] = (current_date - df['reviewTime']).dt.days
df['days'].head()
df['days'].tail()

df.head()
# endregion

# region Time-Based Weighted Average

q1 = df["days"].quantile(0.25)
q2 = df["days"].quantile(0.50)
q3 = df["days"].quantile(0.75)

df.loc[df['days'] <= q1, 'overall'].mean() * 28/100 + \
    df.loc[(df['days'] > q1) & (df['days'] <= q2), 'overall'].mean() * 26/100 + \
    df.loc[(df['days'] > q2) & (df['days'] <= q3), 'overall'].mean() * 24/100 + \
    df.loc[(df['days'] > q3), 'overall'].mean() * 22/100

# endregion

# region Comparison & Evaluation

# when q1 is equal to 280.0, for df['days'] <= q1
df.loc[df['days'] <= q1, 'overall'].mean()  # result is 4.6957928802588995

# when q2 is equal to 430.0, for (df['days'] > q1) & (df['days'] <= q2)
df.loc[(df['days'] > q1) & (df['days'] <= q2), 'overall'].mean()  # result is 4.636140637775961

# when q3 is equal to 600, for (df['days'] > q2) & (df['days'] <= q3)
df.loc[(df['days'] > q2) & (df['days'] <= q3), 'overall'].mean()  # result is 4.571661237785016

# and for df['days'] > q3
df.loc[(df['days'] > q3), 'overall'].mean()  # result is 4.4462540716612375

"""
As a result of comparing the average of the overall product rating according to the time periods
divided by the q1, q2, q3 quarterly values, we can observe that there is an increase in the product
rating from the previous months to the recent date. 

-- For example:
   When the times before the q1 time period are taken into consideration, 
   the product rating average is around 4.69 between these dates, 
   while it is 4.63 in the time period between q1 and q2.

As a result of the increasing interest and satisfaction in the product day by day, 
taking this information into consideration while making a rating calculation means 
offering a more up-to-date evaluation to the users. 
Because, in another scenario, where interest and satisfaction decrease day by day, 
it is possible to encounter a very possible result that the user is not satisfied 
if the weighting is not done according to the time period.

For this reason, the weights that I find appropriate to be given to the time periods 
from the recent date to the more distant dates are:

-- for df['days'] <= q1 --> % 28
-- for (df['days'] > q1) & (df['days'] <= q2) --> % 26
-- for (df['days'] > q2) & (df['days'] <= q3) --> % 24
-- for df['days'] > q3 --> %22 

New rating average value as a result of weighting: 4.59

"""

# endregion

# Determining 20 reviews for the product to be displayed on the product detail page

# region Creating helpful_no variable

df['helpful_no'] = df['total_vote'] - df['helpful_yes']

# endregion

# region Creating score_pos_neg_diff Function & Score


def score_pos_neg_diff(up, down):
    return up - down


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# endregion

# region Creating score_average_rating Function & Score


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


score_average_rating(1952, 2020)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# endregion

# region wilson_lower_bound Function & Score


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


# endregion

# region Top 20 comments by wilson_lower_bound

Comment_Evaluation = pd.DataFrame
Comment_Evaluation = df['reviewText']

Comment_Evaluation = df[['reviewText', 'wilson_lower_bound']].merge(right=Comment_Evaluation,
                                                                    on='reviewText', how='right')
Comment_Evaluation.sort_values(by='wilson_lower_bound', ascending=False).head(20)

# endregion

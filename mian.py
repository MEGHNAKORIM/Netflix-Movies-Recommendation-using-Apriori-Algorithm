import pandas as pd
import numpy as np
from apyori import apriori

# Load the dataset
df = pd.read_csv('NetflixMoviesWatched.csv', header=None, encoding='latin-1')

# Prepare transactions list
transaction_list = []
for row in df.itertuples(index=False):
    transaction = [str(item) for item in row if pd.notnull(item)]
    transaction_list.append(transaction)

# Apply Apriori algorithm
frequent_itemsets = apriori(
    transactions=transaction_list,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    max_length=2
)

# Convert the results into a list
association_rules = list(frequent_itemsets)

# Function to extract relevant information from rules
def extract_rule_info(rule_set):
    lhs_items = []
    rhs_items = []
    support_values = []

    for rule in rule_set:
        for ordered_stat in rule.ordered_statistics:
            if len(ordered_stat.items_base) == 1 and len(ordered_stat.items_add) == 1:
                lhs_items.append(list(ordered_stat.items_base)[0])
                rhs_items.append(list(ordered_stat.items_add)[0])
                support_values.append(rule.support)
                break  # Only take the first rule to avoid duplicates

    return pd.DataFrame({
        'Movie 1': lhs_items,
        'Movie 2': rhs_items,
        'Support': support_values
    })

# Generate the DataFrame and show top 10 associations
result_df = extract_rule_info(association_rules)
top_associations = result_df.nlargest(10, 'Support')
print(top_associations)

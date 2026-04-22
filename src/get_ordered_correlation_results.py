import pandas as pd


# ordered_features = []
# cf = pd.read_csv("spearman_correlation_results_cf.csv")
# cf['abs'] = cf['has_fibrosis'].abs()
# cf = cf.sort_values(by='abs', ascending=False)
# cf = cf.drop(columns=['abs'])
# cf.to_csv("spearman_ordered_correlation_results_cf_label0.csv", index=False)
# ordered_features.append(cf['Unnamed: 0'].tolist()[:20])
# for i in range(1, 7):
#     cf['abs'] = cf[f'point_{i}'].abs()
#     cf = cf.sort_values(by='abs', ascending=False)
#     cf = cf.drop(columns=['abs'])
#     cf.to_csv(f"spearman_ordered_correlation_results_cf_label{i}.csv", index=False)
#     ordered_features.append(cf['Unnamed: 0'].tolist()[:20])

# print(ordered_features)


cf = pd.read_csv("spearman_correlation_results_cf.csv")
cf['abs_sum'] = cf.iloc[:, 1:].abs().sum(axis=1)
cf = cf.sort_values(by='abs_sum', ascending=False)
cf = cf.drop(columns=['abs_sum'])
cf.to_csv("spearman_ordered_correlation_results_cf.csv", index=False)
ordered_features = cf['Unnamed: 0'].tolist()[:20]
print(ordered_features)
import monkdata as m
import dtree as d
import drawtree_qt5 as dt
import random
import numpy as np
import matplotlib.pyplot as plt


# #------------- Assignment 1 -------------------------
# m1_entropy = d.entropy(m.monk1)
# m2_entropy = d.entropy(m.monk2)
# m3_entropy = d.entropy(m.monk3)
# print(f"m1_entropy: {m1_entropy}\nm2_entropy: {m2_entropy}\nm3_entropy: {m3_entropy}\n")

# #------------- Assignment 3 och 4 -------------------------
# datasets = [m.monk1,m.monk2,m.monk3]
# attributes = [0,1,2,3,4,5] #motsvarar a1,a2,...,a6
# for dataset in datasets:
#     for attribute in attributes:
#         print(f"Average gain:{round(d.averageGain(dataset,m.attributes[attribute]),5)} for dataset {datasets.index(dataset)} and attribute {attribute}\n")
# #Bästa attributen att splita monk1 a5 med gain på .287
# #------------- Assignment 5-------------------------

## part 1 analyizing the tree building process
root = m.attributes[4]
subsets = {}
for v in root.values:
    subsets[v] = d.select(m.monk1, root, v)

for value, subset in subsets.items():
    majority = d.mostCommon(subset)
    print(f"a5 = {value}: size={len(subset)}, majority class = {majority}")

remaining_attributes = [a for a in m.attributes if a != root]

for value, subset in subsets.items():
    print(f"\nInformation gain for a5 = {value}")
    for attr in remaining_attributes:
        gain = d.averageGain(subset, attr)
        print(f"  {attr.name}: {gain:.5f}")

tree = d.buildTree(m.monk1, m.attributes, maxdepth=3)
dt.drawTree(tree)

## part 2 testing the datasets
# datasets = [
#     ("MONK-1", m.monk1, m.monk1test),
#     ("MONK-2", m.monk2, m.monk2test),
#     ("MONK-3", m.monk3, m.monk3test),
# ]

# print("Dataset   E_train   E_test")
# print("----------------------------")

# for name, train, test in datasets:
#     tree = d.buildTree(train, m.attributes)
#     train_error = 1 - d.check(tree, train)
#     test_error = 1 - d.check(tree, test)
#     print(f"{name:7s}   {train_error:f}     {test_error:f}")
#------------Assignment 7---------------------------------
# def partition(data, fraction):
#     data = list(data)
#     random.shuffle(data)
#     cut = int(len(data) * fraction)
#     return data[:cut], data[cut:]

# def prune_tree(tree, validation_set):
#     best_tree = tree
#     best_score = d.check(tree, validation_set)

#     improved = True
#     while improved:
#         improved = False
#         for candidate in d.allPruned(best_tree):
#             score = d.check(candidate, validation_set)
#             if score >= best_score:
#                 best_tree = candidate
#                 best_score = score
#                 improved = True
#     return best_tree

# def evaluate_pruning(dataset, testset, fractions, runs=50):
#     results = {f: [] for f in fractions}

#     for f in fractions:
#         for _ in range(runs):
#             train, val = partition(dataset, f)
#             tree = d.buildTree(train, m.attributes)
#             pruned = prune_tree(tree, val)
#             test_error = 1 - d.check(pruned, testset)
#             results[f].append(test_error)

#     return results

# fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# monk1_results = evaluate_pruning(m.monk1, m.monk1test, fractions)
# monk3_results = evaluate_pruning(m.monk3, m.monk3test, fractions)

# print("\nPrinted results (mean ± std):")

# for name, results in [("MONK-1", monk1_results), ("MONK-3", monk3_results)]:
#     print(f"\n{name}")
#     for f in fractions:
#         values = results[f]
#         mean = np.mean(values)
#         std  = np.std(values)
#         print(f"  fraction={f}: mean error={mean:.3f}, std={std:.3f}")


# def plot_results(results, title):
#     means = []
#     stds  = []

#     for f in fractions:
#         values = results[f]
#         means.append(np.mean(values))
#         stds.append(np.std(values))

#     plt.figure()
#     plt.errorbar(
#         fractions,        
#         means,             
#         yerr=stds,         
#         fmt='o-',          
#         capsize=5
#     )

#     plt.xlabel("Training fraction")
#     plt.ylabel("Test classification error")
#     plt.title(title)
#     plt.grid(True)


# plot_results(monk1_results, "MONK-1: Pruning effect")
# plot_results(monk3_results, "MONK-3: Pruning effect")

# plt.show()
"""
This script defines the model used for calculation
"""

from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion='gini'),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion='entropy')
}

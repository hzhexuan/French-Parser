import os
from nltk import Tree

#useful tools
def remove_label(tree):
  for index, subtree in enumerate(tree):
    if type(subtree) != str:
      label = subtree.label().split('-')[0]
      subtree.set_label(label)
      tree[index] = subtree
      remove_label(subtree)
  return tree

def str_to_tree(string):
  tree = Tree.fromstring(string, remove_empty_top_bracketing=True)
  tree = remove_label(tree)
  tree.collapse_unary(collapsePOS=True)
  tree.chomsky_normal_form()
  return tree

def tree_to_str(tree):
  return ' '.join(str(tree).split())

def tree_to_token(tree):
    return tree.leaves()

def load(path):
    #load data
    data = []
    full_path = os.path.join(path)
    with open(full_path) as f:
      for i, l in enumerate(f):
        data.append(l)
    
    #data split
    l = len(data)
    train_data = data[:int(0.8*l)]
    val_data = data[int(0.8*l):int(0.9*l)]
    test_data = data[int(0.9*l):]
    return train_data, val_data, test_data


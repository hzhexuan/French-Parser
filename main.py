from OOV import substitution, load_embeddings, normalize
from tool import load
from PCFG import PCFG
from tqdm import tqdm
from tool import str_to_tree, tree_to_str, tree_to_token
from CYK import CYK
from PYEVALB import scorer

train_data, val_data, test_data = load("sequoia-corpus+fct.mrg_strict")

# PCFG
rules, non_terminal, rules_proba, rules_grouped, lexicon, inverse_map = PCFG(train_data)

# Load embeddings
embeddings_path = "19bSmJm"
words, embeddings = load_embeddings(embeddings_path)

# Map words to indices and vice versa
word_id = {w:i for (i, w) in enumerate(words)}
id_word = dict(enumerate(words))

lexicon_2_lexicon_normalized = {e:normalize(e, word_id) for e in lexicon}
lexicon_normalized_2_lexicon = {}
for k in lexicon_2_lexicon_normalized.keys():
  lexicon_normalized_2_lexicon[lexicon_2_lexicon_normalized[k]] = k
lexicon_normalized_2_lexicon.pop(None)

# Deal with <UNK> with valid set
UNK_rules = {}
count = 0
with tqdm(val_data[:2]) as lines:
    for line in lines:
      t = str_to_tree(line)
      for p in t.productions():
        if(type(p.rhs()[0])==str):
          word = p.rhs()[0]
          if word not in lexicon and substitution(word, word_id, id_word, embeddings, lexicon_normalized_2_lexicon)[0] == "<UNK>":
            count += 1
            if p.lhs() not in UNK_rules:
              UNK_rules[p.lhs()] = 1
            else:
              UNK_rules[p.lhs()] += 1

for k in UNK_rules.keys():
  UNK_rules[k] = UNK_rules[k]/count


def evaluate(dataset, gold_file, test_file):
  with open(gold_file, 'w', encoding="utf8")as f:
    for line in dataset:
      tree = str_to_tree(line)
      tree.un_chomsky_normal_form()
      original = tree_to_str(tree)
      f.write("%s\n" % original)

  with open(test_file, 'w', encoding="utf8")as f:
    with tqdm(dataset) as lines:
        for line in lines:
          tree = str_to_tree(line)
          token = tree_to_token(tree)
          prediction = CYK(token, lexicon, rules, non_terminal, rules_proba, inverse_map, word_id, id_word, embeddings, lexicon_normalized_2_lexicon, UNK_rules)
          prediction_tree = str_to_tree(prediction)
          prediction_tree.un_chomsky_normal_form()
          prediction = tree_to_str(prediction_tree)
          f.write("%s\n" % prediction)

evaluate(test_data[:8], "gold.txt", "evaluation_data.parser_output")

s = scorer.Scorer()
gold_path = 'gold.txt'
test_path = 'evaluation_data.parser_output'
result_path = 'result.txt'

s.evalb(gold_path, test_path, result_path)


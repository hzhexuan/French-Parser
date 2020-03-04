from nltk.grammar import Production
from OOV import substitution

def CYK(token, lexicon, rules, non_terminal, rules_proba, inverse_map, word_id, id_word, embeddings, lexicon_normalized_2_lexicon, UNK_rules):
  p = {}
  records = {}
  # Initialization proba table
  # print("Initialization")
  for i in range(len(token)):
    p[i,i] = {}
    if token[i] in lexicon:
      word = token[i]
      for X in non_terminal.keys():
        if Production(X, (word,)) in rules:
          p[i,i][X] = rules_proba[Production(X, (word,))]
    else:
      words = substitution(token[i], word_id, id_word, embeddings, lexicon_normalized_2_lexicon, k=7)
      #print(i, token[i], words)
      if words[0] == "<UNK>":
        for X in UNK_rules.keys():
          p[i,i][X] = UNK_rules[X]
      else:
        for word in words:
          for X in non_terminal.keys():
            if Production(X, (word,)) in rules:
              p[i,i][X] = rules_proba[Production(X, (word,))] if X not in p[i,i].keys() else p[i,i][X] + rules_proba[Production(X, (word,))]
  #print(p)
  # print("Construction")
  for l in range(1, len(token)):
    for i in range(len(token)-l):
      j = i + l
      p[i,j] = {}
      for k in range(i,j):
        H1 = p[i,k]
        H2 = p[k+1,j]
        for e1 in H1.keys():
          for e2 in H2.keys():
            if((e1,e2) in inverse_map.keys()):
              parents = inverse_map[(e1,e2)]
              for parent in parents:
                score = rules_proba[Production(parent,(e1,e2))] * H1[e1] * H2[e2]
                if(parent not in p[i,j] or p[i,j][parent] < score):
                  p[i,j][parent] = score
                  records[i,j,parent] = e1, e2, k
  #print(p[0, len(token) - 1])
  # print("time:", end-start)
  # Get tree from records:
  def get_tree_from_records(i,j,X):
    if(i == j):
      return "".join([str(X), ' ', str(token[i])])
    Y,Z,s = records[i,j,X]
    return "".join([str(X), "(", get_tree_from_records(i,s,Y), ")(", get_tree_from_records(s+1,j,Z), ")"])

  # Retrieve the most probable parsed tree from backpointers and argmax of probability table
  # print("Retrieve")
  max_score = 0
  X_best = None
  for X in non_terminal.keys():
    if X in p[0, len(token) - 1].keys() and max_score < p[0, len(token) - 1][X]:
      max_score = p[0, len(token) - 1][X]
      X_best = X
  if X_best == None:
    return '(SENT (UNK))'
  else:
    return '(SENT (' + get_tree_from_records(0,len(token)-1,X_best) + '))'
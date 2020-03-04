import nltk
from tool import str_to_tree

def PCFG(train_data):
    #PCFG
    rules = {}
    non_terminal = {}
    rules_proba = {}
    rules_grouped = {}
    inverse_map = {}
    for string in train_data:
      tree = str_to_tree(string)
      for p in tree.productions():
        if p.lhs() == nltk.grammar.Nonterminal("SENT"):
          #print(p)
          continue
        if p in rules:
          rules[p] += 1
        else:
          rules[p] = 1
        
        if p.lhs() in non_terminal:
          non_terminal[p.lhs()] += 1
        else:
          non_terminal[p.lhs()] = 1
    
        if p.lhs() not in rules_grouped.keys():
          rules_grouped[p.lhs()] = []
        rules_grouped[p.lhs()].append(p.rhs())
    
        if p.rhs() not in inverse_map.keys():
          inverse_map[p.rhs()] = []
        
        inverse_map[p.rhs()].append(p.lhs())
    
    for rule in rules.keys():
      rules_proba[rule] = rules[rule]/non_terminal[rule.lhs()]
    
    lexicon = [symbol.rhs()[0] for symbol in rules.keys() if type(symbol.rhs()[0])==str]

    return rules, non_terminal, rules_proba, rules_grouped, lexicon, inverse_map
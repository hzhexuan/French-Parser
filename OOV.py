import re
import pickle
from operator import itemgetter

#OOV
def load_embeddings(path):
  with open(path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    words, embeddings = u.load()
  return words, embeddings

def get_neibor(word_list):
  S = "abcdefghijklmnopqrstuvwxyzàâôéèëêïîçùœ_-'/"
  neibors = []

  for word in word_list:
    for i in range(len(word)):
      e = word[:i]+word[i+1:] #delete
      neibors.append(e) #delete
      for c in S:
        e = word[:i]+c+word[i+1:] #substitution
        neibors.append(e)
        e = word[:i]+c+word[i:] #insert
        neibors.append(e)
    for c in S:
      e = word+c #end insert
      neibors.append(e)

  return neibors

def k_neibors(word, k=2):
  s = set()
  s.add(word)
  l = set(get_neibor(s))
  neibors = l.copy()
  for i in range(1,k):
    l = set(get_neibor(l-s))
    s = neibors.copy()
    neibors = neibors|l

  if word in neibors:
    neibors.remove(word)
  return neibors

def case_normalizer(word, dictionary):
  """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
  w = word
  lower = (dictionary.get(w.lower(), 1e12), w.lower())
  upper = (dictionary.get(w.upper(), 1e12), w.upper())
  title = (dictionary.get(w.title(), 1e12), w.title())
  results = [lower, upper, title]
  results.sort()
  index, w = results[0]
  if index != 1e12:
    return w
  return word


def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    DIGITS = re.compile("[0-9]", re.UNICODE)
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def l2_nearest(embeddings, word_index):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

    e = embeddings[word_index]
    distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances)


def knn(word, embeddings, word_id, id_word):
  original_word = word
  word = normalize(word, word_id)
  if word:
    word_index = word_id[word]
    indices, distances = l2_nearest(embeddings, word_index)
    neighbors = [id_word[idx] for idx in indices]
    return list(zip(neighbors, distances))
  else:
    L = []
    neibors = k_neibors(original_word, k=1)
    for e in neibors:
      e = normalize(e, word_id)
      if not e:
        continue
      word_index = word_id[e]
      indices, distances = l2_nearest(embeddings, word_index)
      neighbors = [id_word[idx] for idx in indices]
      L += list(zip(neighbors, distances))
    if(len(L) == 0):
      neibors = k_neibors(original_word, k=2)
      for e in neibors:
        e = normalize(e, word_id)
        if not e:
          continue
        word_index = word_id[e]
        indices, distances = l2_nearest(embeddings, word_index)
        neighbors = [id_word[idx] for idx in indices]
        L += list(zip(neighbors, distances))        
    return sorted(L, key=lambda x:x[1])


def substitution(word, word_id, id_word, embeddings, lexicon_normalized_2_lexicon, k=7):
  word_normalized = normalize(word, word_id)
  if word_normalized in lexicon_normalized_2_lexicon.keys():
    return [lexicon_normalized_2_lexicon[word_normalized]]

  L = knn(word, embeddings, word_id, id_word)
  result = []
  for (neibor,_) in L:
    if neibor in lexicon_normalized_2_lexicon.keys():
      result.append(lexicon_normalized_2_lexicon[neibor])
    if len(result) == k:
      return result
  return ["<UNK>"] if len(result) == 0 else result

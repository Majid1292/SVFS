import pandas as pd
import numpy as np
import networkx as nx
import random

from mutual_info import compute_mi


class svfs:
  round_threshold=3
  def __init__(self,x_train,class_label,x_threshold=3,diff_threshold=1.7,mean_threshold=4,n_feature_threshold=50,alpha=50,beta=5):
      self.__A=x_train
      self.__b=class_label
      self.__x_threshold=x_threshold
      self.__diff_threshold = diff_threshold
      self.__mean_threshold = mean_threshold
      self.__n_feature_threshold = n_feature_threshold
      self.__chunk_threshold = alpha
      self.__short_chunk_threshold = beta

  def high_rank_x(self):
      iA = np.linalg.pinv(self.__A)  # Computes the (Moore-Penrose) pseudo-inverse of a matrix
      # Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD)
      X = np.matmul(iA, self.__b)  # Matrix product of two arrays
      abs_X = abs(X)
      mean = np.mean(np.array(abs_X), axis=0)
      high_x, = np.where(np.array(abs_X) > mean * (self.__x_threshold - 1))
      return high_x

  def reduction(self):
      iA = np.linalg.pinv(self.__A)
      X = np.matmul(iA,self.__b)
      mean = np.mean(np.array(abs(X)), axis=0)
      list_highX, = np.where(np.array(abs(X)) > mean * (self.__x_threshold - 1))
      list_highX1, = np.where(np.array(abs(X)) > mean * self.__x_threshold)
      diff_lenX = list_highX.shape[0] - list_highX1.shape[0]
      A = self.__A.iloc[:, np.array(list_highX)]
      num_features = A.shape[1]
      B = np.array(self.__b)[np.newaxis]
      AB = np.append(A, B.T, 1)
      iAB = np.linalg.pinv(AB)
      identity_matrix = np.identity(num_features + 1)
      S_AB = identity_matrix - (np.mat(iAB) * np.mat(AB))
      ss = pd.DataFrame(S_AB)
      round_S = pd.DataFrame(abs(np.array(S_AB).round(svfs.round_threshold)))
      outliers_idx = []
      outliers_val = []
      S_colunm = abs(ss.copy().iloc[:, ss.shape[1] - 1])

      sort_s_column = np.array(np.argsort(S_colunm)[::-1])

      sort_idx = np.delete(sort_s_column, 0)
      sort_idx = sort_idx[:len(sort_idx) - int(diff_lenX * self.__diff_threshold)]
      return sort_idx

  def selection(self):
      list_highX = svfs.high_rank_x(self)
      sort_idx = svfs.reduction(self)
      clean_features = list_highX[sort_idx]
      A_clean = self.__A[clean_features]
      num_features = A_clean.shape[1]
      iA = np.linalg.pinv(A_clean)
      S_A = abs(np.identity(num_features) - (np.mat(iA) * np.mat(A_clean)))
      df_S_A = pd.DataFrame(S_A)
      diag_s = np.array(S_A).diagonal()
      sort_diag = np.array(np.argsort(diag_s)[::-1])
      clusters = []
      G = nx.DiGraph()
      for index, value in enumerate(diag_s):
          high_s, = np.where(df_S_A.iloc[:, index] > self.__mean_threshold * np.mean(df_S_A.iloc[:, index]))
          if not high_s.all():
              pass
          else:
              for high_index, high_value in enumerate(high_s):
                  G.add_node(index, visited=False)
                  G.add_node(high_value, visited=False)
                  G.add_edge(index, high_value)

      nodes = nx.nodes(G)
      select_cl = None
      select_cls = []
      dic_cls = {}
      rnd = 0
      mi_info = 0
      y = self.__b.copy()
      while len(nodes) > 0 and len(dic_cls) < self.__n_feature_threshold:
          root = np.array(nodes)[rnd]
          edges = list(nx.bfs_edges(G, root))
          if edges:
              cluster = np.unique(np.array(edges).flatten())
              cluster = np.setdiff1d(cluster, clusters)
              clusters.extend(cluster)
              max_mi = 0
              if len(cluster) > self.__chunk_threshold:
                  random.shuffle(cluster)
                  chunks = [cluster[x:x + self.__chunk_threshold] for x in range(0, len(cluster), self.__chunk_threshold)]
                  for chunk in enumerate(chunks):
                      locs_ab = []
                      for index, value in enumerate(chunk[1]):
                          loc_ab, = np.where(clean_features[value] == list_highX)
                          locs_ab.extend(loc_ab)
                      short_chunk = sorted(locs_ab, key=list(sort_idx).index)
                      for index, value in enumerate(np.array(short_chunk[0:self.__short_chunk_threshold - 1])):
                          x = self.__A.iloc[:, list_highX[value]].copy()
                          mi_info = compute_mi(x, y)
                          if mi_info > max_mi:
                              max_mi = mi_info
                              select_cl, = np.where(clean_features == list_highX[value])
                      select_cls.append(select_cl[0])
                      dic_cls[select_cl[0]] = max_mi
                      max_mi = 0
                  nodes = np.setdiff1d(nodes, cluster)
              else:
                  if len(cluster) == 1:
                      dic_cls[cluster[0]] = mi_info
                      nodes = np.delete(nodes, rnd)
                  else:
                      for index, value in enumerate(cluster):
                          x = self.__A.iloc[:, clean_features[value]].copy()
                          mi_info = compute_mi(x, y)
                          if mi_info > max_mi:
                              max_mi = mi_info
                              select_cl = value
                      select_cls.append(select_cl)
                      dic_cls[select_cl] = max_mi
                      nodes = np.setdiff1d(nodes, cluster)
          else:
              dic_cls[root] = mi_info
              nodes = np.delete(nodes, rnd)
      return  dic_cls


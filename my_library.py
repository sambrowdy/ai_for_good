def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

  #your function body below - copy and paste then align with parameter names
  t_subset = up_table_subset(full_table, the_target_column, 'equals', the_target_column_value)
  e_list = up_get_column(t_subset, the_evidence_column)
  p_b_a = sum([1 if v==the_evidence_column_value else 0 for v in e_list])/len(e_list)
  return p_b_a

def cond_probs_product(full_table, evidence_row, target_column, target_column_value):
  assert target_column in full_table
  assert target_column_value in up_get_column(full_table, target_column)
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target column from full_table

  #your function body below
  evidence_columns = up_list_column_names(full_table)[:-1]
  evidence_complete = list(zip(evidence_columns, evidence_row))
  cond_prob_list = [cond_prob(full_table, evidence_column, evidence_value, target_column, target_column_value) for evidence_column, evidence_value in evidence_complete]
  return up_product(cond_prob_list)

def prior_prob(full_table, the_column, the_column_value):
  assert the_column in full_table
  assert the_column_value in up_get_column(full_table, the_column)

  #your function body below
  t_list = up_get_column(full_table, the_column)
  p_a = sum([1 if v==the_column_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(full_table, evidence_row, target_column):
  assert target_column in full_table
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target

  #compute P(target=0|...) by using cond_probs_product, finally multiply by P(target=0) using prior_prob
  neg = cond_probs_product(full_table, evidence_row, target_column, 0) * prior_prob(full_table, target_column, 0)

  #do same for P(target=1|...)
  pos = cond_probs_product(full_table, evidence_row, target_column, 1) * prior_prob(full_table, target_column, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(neg, pos)
  #return your 2 results in a list
  return [neg, pos]

def metrics(zipped_list):
  assert isinstance(zipped_list, list)
  assert all([isinstance(v, list) for v in zipped_list])
  assert all([len(v)==2 for v in zipped_list])
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  #first compute the sum of all 4 cases. See code above
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision = round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0
  recall = 0 if tp + fn == 0 else round(tp / (tp + fn), 2)
  f1 = 0 if precision + recall == 0 else round(2 * (precision * recall) / (precision + recall), 2)
  accuracy = round((tp + tn) / (tp + tn + fp + fn), 2) if (tp + tn + fp + fn) != 0 else 0

  #now build dictionary with the 4 measures - round values to 2 places
  results = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

  #finally, return the dictionary
  return results

from sklearn.ensemble import RandomForestClassifier

def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use

  assert target in train   #have not dropped it yet
  assert target in test

  #your code below - copy, paste and align from above
  k_feature_table = up_drop_column(train, target)
  k_actuals = up_get_column(train, target)

  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0) 
  clf.fit(k_feature_table, k_actuals)

  probs = clf.predict_proba(up_drop_column(test,target))  #Note no need here to transform k_feature_table to list - we can just use the table. Nice.

  pos_probs = [p for n,p in probs]  #just the positive probabilities

  all_mets = []
  for t in thresholds:
    predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(predictions, up_get_column(test, target))
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]
  metrics_table = up_metrics_table(all_mets)
  return metrics_table

def try_archs(train_table, test_table, target_column_name, architectures, thresholds):
  arch_acc_dict = {}  #ignore if not attempting extra credit

  #now loop through architectures
  for arch in architectures:
    probs = up_neural_net(train_table, test_table, arch, target_column_name)
    pos_probs = [pos for neg,pos in probs]

    #loop through thresholds
    all_mets = []
    for t in thresholds:
      predictions = [1 if pos>=t else 0 for pos in pos_probs]
      pred_act_list = up_zip_lists(predictions, up_get_column(test_table, target_column_name))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    arch_acc_dict[tuple(arch)] = max([metd['Accuracy'] for metd in all_mets])  #extra credit - uncomment if want to attempt

    print(f'Architecture: {arch}')
    display(up_metrics_table(all_mets))

  return arch_acc_dict

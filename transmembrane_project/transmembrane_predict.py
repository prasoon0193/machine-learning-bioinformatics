import numpy as np
import argparse
import hmm_jointprob_transmembrane as hmm_jp
import re

np.seterr(divide='ignore')

hidden = {}
observables = {}
rev_observables = {}
pi = []
transitions = []
emissions = []

def train_hmm(model, specific_files=None):
  global hidden, observables, rev_observables, pi, transitions, emissions

  hidden = {}
  observables = {}
  rev_observables = {}
  pi = []
  transitions = []
  emissions = []

  # pi, transitions & emissions will be all zeros initially in 3-state
  if model == '3-state':
    print("TRAINING 3-STATE")
    # initialize hidden
    str_hidden = 'i M o'
    for (i, v) in enumerate(str_hidden.split(' ')):
      hidden[i] = v    
  
    # initialize observables
    str_observables = 'A C E D G F I H K M L N Q P S R T W V Y'
    for (i, v) in enumerate(str_observables.split(' ')):
      observables[i] = v

    # reverse key:value for compatibility with viterbi_logspace_backtrack
    rev_observables = {v: k for k, v in observables.items()}

    # initialize pi; K x 1
    for _ in range(len(hidden)):
      pi.append(0.0)

    # initialize transitions; K x K
    K = len(hidden); # K = number of hidden states
    for i in range(K):
      transitions.append([])
      for j in range(K):
        transitions[i].append(0.0)
    transitions = np.array(transitions)

    # initialize emissions; K x Obs where Obs is number of observables
    Obs = len(observables); # K = number of hidden states
    for i in range(K):
      emissions.append([])
      for j in range(Obs):
        emissions[i].append(0.0)
    emissions = np.array(emissions)

    # retrieve event counting for training
    total_transitions, trans_event_countings, total_emissions, \
    emission_event_countings, init_states = count_events_in_data_files('3-state', specific_files)

    # normalize countings
    for k, v in trans_event_countings.items():
      trans_event_countings[k] = v / total_transitions[k[0]]

    for k, v in emission_event_countings.items():
      emission_event_countings[k] = v / total_emissions[k[0]]
      # emission_event_countings[k] = float("{0:.5f}".format(v / total_emissions[k[0]]))

    total_sequences = sum(init_states.values())
    for k, v in init_states.items():
      init_states[k] = v / total_sequences

    # update hmm with probabilities based on event countings
    for k, v in trans_event_countings.items():
      idx_from = hidden_to_idx(k[0])
      idx_to = hidden_to_idx(k[1])
      transitions[idx_from][idx_to] = v

    for k, v in emission_event_countings.items():
      idx_hidden = hidden_to_idx(k[0])
      idx_obs = obs_to_idx(k[1])
      emissions[idx_hidden][idx_obs] = v

    for k, v in init_states.items():
      pi[hidden_to_idx(k)] = v

  # o -> N -> i -> M -> o
  elif model == '4-state':
    print("TRAINING 4-STATE")
    # initialize hidden
    str_hidden = 'i M N o'
    for (i, v) in enumerate(str_hidden.split(' ')):
      hidden[i] = v    
  
    # initialize observables
    str_observables = 'A C E D G F I H K M L N Q P S R T W V Y'
    for (i, v) in enumerate(str_observables.split(' ')):
      observables[i] = v

    # reverse key:value for compatibility with viterbi_logspace_backtrack
    rev_observables = {v: k for k, v in observables.items()}

    # initialize pi; K x 1
    for _ in range(len(hidden)):
      pi.append(0.0)

    # initialize transitions; K x K
    K = len(hidden); # K = number of hidden states
    for i in range(K):
      transitions.append([])
      for j in range(K):
        transitions[i].append(0.0)
    transitions = np.array(transitions)

    # initialize emissions; K x Obs where Obs is number of observables
    Obs = len(observables); # K = number of hidden states
    for i in range(K):
      emissions.append([])
      for j in range(Obs):
        emissions[i].append(0.0)
    emissions = np.array(emissions)

    # retrieve event counting for training
    total_transitions, trans_event_countings, total_emissions, \
    emission_event_countings, init_states = count_events_in_data_files('4-state', specific_files)

    # normalize countings
    for k, v in trans_event_countings.items():
      trans_event_countings[k] = v / total_transitions[k[0]]

    for k, v in emission_event_countings.items():
      emission_event_countings[k] = v / total_emissions[k[0]]
      # emission_event_countings[k] = float("{0:.5f}".format(v / total_emissions[k[0]]))

    total_sequences = sum(init_states.values())
    for k, v in init_states.items():
      init_states[k] = v / total_sequences

    # update hmm with probabilities based on event countings
    for k, v in trans_event_countings.items():
      idx_from = hidden_to_idx(k[0])
      idx_to = hidden_to_idx(k[1])
      transitions[idx_from][idx_to] = v

    for k, v in emission_event_countings.items():
      idx_hidden = hidden_to_idx(k[0])
      idx_obs = obs_to_idx(k[1])
      emissions[idx_hidden][idx_obs] = v

    for k, v in init_states.items():
      pi[hidden_to_idx(k)] = v

def obs_to_idx(argObs):
  for (k, v) in observables.items():
    if v == argObs:
      return k
  return None

def hidden_to_idx(argHiddenState):
  for (k, v) in hidden.items():
    if v == argHiddenState:
      return k
  return None  

def count_single_sequence_4state(argObsSeq, argHiddenSeq):
  def replace_Ns(inputseq):
    """ given a sequence of hidden states, replaces the occurences of 'M'
        corresponding to the state 'N' in the 4-state model with the actual
        letter 'N' to ease event counting """
    replace_indices = []

    for match in re.finditer('M+i', inputseq):
      replace_indices.append(match.span())
    result = list(inputseq)
    for repl in replace_indices:
      idx_start = repl[0]
      idx_end = repl[1] - 1
      result[idx_start:idx_end] = 'N' * (idx_end - idx_start)
    intermediate_result = ''.join(result)

    replace_indices.clear()
    result.clear()

    for match in re.finditer('oM+', intermediate_result):
      replace_indices.append(match.span())
    result = list(intermediate_result)
    for repl in replace_indices:
      idx_start = repl[0] + 1
      idx_end = repl[1]
      result[idx_start:idx_end] = 'N' * (idx_end - idx_start)
    final_result = ''.join(result)

    return final_result

  seq_transitions = {}
  seq_emissions = {}
  nr_transitions = {}
  nr_emissions = {}
  init_state = {}
  if (argHiddenSeq[0] in ['i', 'o']):
    init_state[argHiddenSeq[0]] = 1
  else:
    # keep going till we see either an 'i' or 'o' to determine whether
    # we're in the membrane going out or in, to decide between 'N' and 'M'
    curr_sym = argHiddenSeq[0]
    c = 0
    while (curr_sym not in ['i', 'o']):
      c += 1
      curr_sym = argHiddenSeq[c]
    if (curr_sym == 'i'):
      # we went from membrane to inside, initial state must have been 'N'
      init_state['N'] = 1
    else: # curr_sym == 'o'
      # we went from membrane to outside, initial state must have been 'M'
      init_state['M'] = 1
  
  argHiddenSeq_N_replaced = replace_Ns(argHiddenSeq)

  nr_emissions, seq_emissions, nr_transitions, seq_transitions, \
  _ = count_single_sequence(argObsSeq, argHiddenSeq_N_replaced)

  return nr_emissions, seq_emissions, nr_transitions, \
         seq_transitions, init_state

def count_single_sequence(argObsSeq, argHiddenSeq):
  seq_transitions = {}
  seq_emissions = {}
  nr_emissions = {}
  nr_transitions = {}

  init_state = {argHiddenSeq[0]: 1}
  
  for n in range(len(argObsSeq) - 1):
    this_obs = argObsSeq[n]
    # next_obs = argObsSeq[n + 1]
    this_hidden = argHiddenSeq[n]
    next_hidden = argHiddenSeq[n + 1]

    # count transitions
    curr_transition = this_hidden + next_hidden
    if curr_transition in seq_transitions:
      seq_transitions[curr_transition] += 1
    else:
      seq_transitions[curr_transition] = 1

    # count emissions
    curr_emission = this_hidden + this_obs
    if curr_emission in seq_emissions:
      seq_emissions[curr_emission] += 1
    else:
      seq_emissions[curr_emission] = 1

    # count totals
    if this_hidden in nr_transitions:
      nr_transitions[this_hidden] += 1
    else:
      nr_transitions[this_hidden] = 1

    if this_hidden in nr_emissions:
      nr_emissions[this_hidden] += 1
    else:
      nr_emissions[this_hidden] = 1

  # include last emission
  last_obs = argObsSeq[len(argObsSeq) - 1]
  last_hidden = argHiddenSeq[len(argHiddenSeq) - 1]
  last_emission = last_hidden + last_obs
  if last_emission in seq_emissions:
    seq_emissions[last_emission] += 1
  else:
    seq_emissions[last_emission] = 1

  if last_hidden in nr_emissions:
    nr_emissions[last_hidden] += 1
  else:
    nr_emissions[last_hidden] = 1

  return nr_emissions, seq_emissions, nr_transitions, seq_transitions, init_state
  
def add_dicts(dict1, dict2):
  """ Returns a dict consisting of all keys from both dict1 and dict2.
      In case of common keys, adds together the corresponding values"""
  result = {}
  for k1, v1 in dict1.items():
    if k1 not in dict2:
      result[k1] = v1
    else:
      result[k1] = v1 + dict2[k1]
  for k2, v2 in dict2.items():
    if k2 not in dict1:
      result[k2] = v2
  return result

def process_sequencefile(argfile, model):
  with open(argfile, 'r') as f:
    
    curr_line = f.readline()
    
    # initialize processing results to empty values
    trans_event_countings = {}
    emission_event_countings = {}
    total_transitions = {}
    total_emissions = {}
    init_states = {}

    # stops when whole sequence file has been processed 
    #  (reaches an empty line where a name was expected)
    while curr_line.strip():

      # load current sequence information from file
      name = curr_line[1:].rstrip()
      observed_seq = f.readline().strip()
      hidden_seq = f.readline()[2:].strip()

      # count transitions & emissions in current sequence
      if (model == '3-state'):
        nr_em, seq_em, nr_trans, \
        seq_trans, init_state = count_single_sequence(observed_seq, hidden_seq)
      else:
        nr_em, seq_em, nr_trans, \
        seq_trans, init_state = count_single_sequence_4state(observed_seq, hidden_seq)
    
      # update result dicts with the countings from this sequence
      trans_event_countings = add_dicts(trans_event_countings, seq_trans)
      emission_event_countings = add_dicts(emission_event_countings, seq_em)
      total_transitions = add_dicts(total_transitions, nr_trans)
      total_emissions = add_dicts(total_emissions, nr_em)
      init_states = add_dicts(init_states, init_state)

      f.readline() # skip empty line
      curr_line = f.readline()

  return total_transitions, trans_event_countings, \
         total_emissions, emission_event_countings, init_states

def count_events_in_data_files(model, specific_files=None):
  # initialize processing results to empty values
  trans_event_countings = {}
  emission_event_countings = {}
  total_transitions = {}
  total_emissions = {}
  total_init_states = {}

  # iterate all ten data files, or specific ones if specified
  file_ids = specific_files if specific_files else range(10)
  for i in file_ids:
    curr_file = 'Dataset160/set160.%i.labels.txt' % i

    # print("TRAINING ON " + curr_file)

    with open(curr_file, 'r') as f:
      curr_total_trans, curr_trans_event_countings, curr_total_em, \
      curr_em_event_countings, curr_init_states = process_sequencefile(curr_file, model)

      # update results with current file countings
      trans_event_countings = add_dicts(trans_event_countings, curr_trans_event_countings)
      emission_event_countings = add_dicts(emission_event_countings, curr_em_event_countings)
      total_transitions = add_dicts(total_transitions, curr_total_trans)
      total_emissions = add_dicts(total_emissions, curr_total_em)
      total_init_states = add_dicts(total_init_states, curr_init_states)

  return total_transitions, trans_event_countings, \
         total_emissions, emission_event_countings, total_init_states

def print_hmm():
  print('hidden')
  print(hidden)

  print('\nobservables')
  print(observables)

  print('\npi')
  print(pi)

  print('\ntransitions')
  print(transitions)

  print('\nemissions')
  print(emissions)

def write_hmm(argfile):
  with open(argfile, 'w') as output:
    # write 'hidden' to file
    output.write('#\n#\n#\n\nhidden\n')
    str_hidden = ''
    for k, v in sorted(hidden.items(), key=lambda x: x[0]):
      str_hidden += v + ' '
    output.write(str_hidden.rstrip())

    # write 'observables' to file
    output.write('\n\nobservables\n')
    str_observables = ''
    for k, v in sorted(observables.items(), key=lambda x: x[0]):
      str_observables += v + ' '
    output.write(str_observables.rstrip())

    # write 'pi' to file
    output.write('\n\npi\n')
    str_pi = ''
    for fl in pi:
      str_pi += str(fl) + ' '
    output.write(str_pi.rstrip())

    # write 'transitions' to file
    output.write('\n\ntransitions\n')
    str_transitions = ''
    for i in range(len(transitions)):
      for j in range(len(transitions[i])):
        str_transitions += str(transitions[i,j]) + ' '
      str_transitions = str_transitions.rstrip() + '\n'
    output.write(str_transitions)

    # write 'emissions' to file
    output.write('\nemissions\n')
    str_emissions = ''
    for i in range(len(emissions)):
      for j in range(len(emissions[i])):
        str_emissions += str(emissions[i,j]) + ' '
      str_emissions = str_emissions.rstrip() + '\n'
    output.write(str_emissions+'\n')

def ten_fold_cross_validation(model):
  file_ids = list(range(10))
  file_ids_to_be_excluded = list(range(10))

  hmm_filename = 'tenfold_current_hmm.txt'

  while file_ids_to_be_excluded: # terminates when list is empty
    # pull one file out for validation
    validation_id = file_ids_to_be_excluded.pop()
    validation_file = 'Dataset160/set160.%i.labels.txt' % validation_id
    # filter the validation file from the list of all files, leaving the 9 others
    training_files = file_ids[: validation_id] + file_ids[validation_id+1 :]

    # train on the 9 remaining files
    train_hmm(model, training_files)


    write_hmm(hmm_filename)
    hmm_jp.load_hmm(hmm_filename)

    # the two should match
    # print_hmm()
    # hmm_jp.print_hmm()

    # predict sequences in validation file, write predictions to output file
    with open(validation_file, 'r') as f, open(validation_file[:-4]+'_PREDICTIONS.txt', 'w') as output:
      curr_line = f.readline()

      # stops when whole sequence file has been processed 
      #  (reaches an empty line where a name was expected)
      while curr_line.strip():
        # load current sequence information from file
        name = curr_line[1:].rstrip()
        observed_seq = f.readline().strip()
        hidden_seq = f.readline()[2:].strip()

        print("RUNNING VITERBI DECODING")
        vit_log_prob, vit_hidden_pred = hmm_jp.viterbi_logspace_backtrack(observed_seq)


        # if we predicted using the 4-state model, the predictions will likely
        # contain instances of 'N', a state in our model corresponding to the
        # actual state 'M'. We need these replaced with 'M' in our final answer
        vit_hidden_pred = re.sub('N', 'M', vit_hidden_pred)

        output.write('>'+name+'\n')
        output.write(' '+observed_seq+'\n')
        output.write('# '+vit_hidden_pred+'\n\n')
        
        # prepare for next sequence
        f.readline() # skip empty line
        curr_line = f.readline()

    print("FINISHED TRAINING\n\n")

def decode_all_data_files(hmmfile):
  hmm_jp.load_hmm(hmmfile)

  for i in range(10):
    curr_file = 'Dataset160/set160.%i.labels.txt' % i

    with open(curr_file, 'r') as f:
      curr_line = f.readline()

      # stops when whole sequence file has been processed 
      #  (reaches an empty line where a name was expected)
      while curr_line.strip():
        # load current sequence information from file
        name = curr_line[1:].rstrip()
        observed_seq = f.readline().strip()
        hidden_seq = f.readline()[2:].strip()

        vit_log_prob, vit_hidden_pred = hmm_jp.viterbi_logspace_backtrack(observed_seq)
        post_log_prob, post_hidden_pred = hmm_jp.posterior_sequence_decoding(observed_seq)

        # if we predicted using the 4-state model, the predictions will likely
        # contain instances of 'N', a state in our model corresponding to the
        # actual state 'M'. We need these replaced with 'M' in our final answer
        vit_hidden_pred = re.sub('N', 'M', vit_hidden_pred)
        post_hidden_pred = re.sub('N', 'M', post_hidden_pred)

        # report predictions
        print("NAME: " + name)
        print("ACTUAL:")
        print(hidden_seq)
        print("VITERBI DECODING:")
        print(vit_hidden_pred)
        print(vit_log_prob)
        print('POSTERIOR DECODING:')
        print(post_hidden_pred)
        print(post_log_prob)
        print('\n\n')

        # prepare for next sequence
        f.readline() # skip empty line
        curr_line = f.readline()
    
def main():

  # train_hmm('3-state')
  # print_hmm()
  # write_hmm('3-state-hmm.txt')
  # train_hmm('4-state')
  # print_hmm()
  # write_hmm('4-state-hmm.txt')

  # decode_all_data_files('3-state-hmm.txt')
  # decode_all_data_files('4-state-hmm.txt')

  ten_fold_cross_validation('4-state')

if __name__ == '__main__':
  main()

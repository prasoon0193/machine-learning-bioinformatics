import numpy as np
import argparse

np.seterr(divide='ignore')

hidden = {}
observables = {}
rev_observables = {}
pi = []
transitions = []
emissions = []

def load_hmm(state):
  global hidden, observables, rev_observables, pi, transitions, emissions

  # pi, transitions & emissions will be all zeros initially in 3-state
  if state == '3-state':

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
    emission_event_countings, init_states = count_events_all_data_files()

    # normalize countings
    for k, v in trans_event_countings.items():
      trans_event_countings[k] = v / total_transitions[k[0]]

    for k, v in emission_event_countings.items():
      emission_event_countings[k] = v / total_emissions[k[0]]

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

def viterbi_logspace_backtrack(aSequence):
  # BASIS
  idx_first_obs = rev_observables.get(aSequence[0])
  omega = np.log([pi]) + np.log(emissions[:,idx_first_obs])
  
  # RECURSIVE
  for obs in range(1, len(aSequence)):

    max_vector = []
    # iterating through all states to generate next col in omega
    for i, _ in enumerate(hidden):

      # find transition probabilities from every state to this current state
      trans_to_state_i = transitions[:,i]
      
      # fetch previous col in omega
      prev_omega_col = omega[-1]

      # find the max probability that this state will follow from the prev col
      state_i_max_prob = np.max(prev_omega_col + np.log(trans_to_state_i))

      # save for multiplying with emission probabilities to determine omega col
      max_vector.append(state_i_max_prob)

    # get idx of current observation to use with defined matrix data structures
    idx_curr_obs = rev_observables.get(aSequence[obs])
    
    # get emission probabilities of current observation for all states
    emissions_curr_obs = emissions[:,idx_curr_obs]
    
    # create and add the new col to the omega table
    new_omega_col = np.log(emissions_curr_obs) + max_vector
    omega = np.append(omega, [new_omega_col], axis=0)

  # natural log to the most likely probability when all the input is processeds
  log_most_likely_prob = np.max(omega[-1])

  # BACKTRACKING

  N = len(aSequence)-1  # off-by-one correction for indexing into lists
  K = len(hidden)
  z = np.zeros(len(aSequence))
  z[N] = np.argmax(omega[len(omega)-1], axis=0)

  # n descending from N-1 to 0 inclusive
  for n in range(N-1, -1, -1):
    max_vector = []
    for k in range(0, K):
      # only for matching pseudocode easily
      x = aSequence

      # matrix data structure index of observation
      idx_obs = rev_observables.get(x[n+1])

      # probability of observing x[n+1] in state z[n+1]
      p_xn1_zn1 = emissions[z[n+1]][idx_obs]

      # our omega table indexing is flipped compared to the pseudocode alg.
      omega_kn = omega[n][k]
 
      # get transitions from state k to state z[n+1]
      p_zn1_k = transitions[k,z[n+1]]

      # add product to max_vector
      max_vector.append(np.log(p_xn1_zn1) + omega_kn + np.log(p_zn1_k))
      
    # set z[n] to arg max of max_vector
    z[n] = np.argmax(max_vector)

  # add one to correspond to actual states rather than indexes into 'states'
  z = z + 1
  
  # conversion from indices to actual state names
  hidden_seq = ""
  for i in z:
    hidden_seq += hidden[i - 1]

  return (log_most_likely_prob, hidden_seq)

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


def process_sequencefile(argfile):
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
      nr_em, seq_em, nr_trans, \
      seq_trans, init_state = count_single_sequence(observed_seq, hidden_seq)
    
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

def count_events_all_data_files():
# initialize processing results to empty values
  trans_event_countings = {}
  emission_event_countings = {}
  total_transitions = {}
  total_emissions = {}
  total_init_states = {}

  # iterate all ten data files
  for i in range(10):
    curr_file = 'Dataset160/set160.%i.labels.txt' % i

    with open(curr_file, 'r') as f:
      curr_total_trans, curr_trans_event_countings, curr_total_em, \
      curr_em_event_countings, curr_init_states = process_sequencefile(curr_file)

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

def main():
  load_hmm('3-state')
  print_hmm()


if __name__ == '__main__':
  main()

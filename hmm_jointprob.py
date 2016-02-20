import numpy as np
import argparse

np.seterr(divide='ignore')

hidden = {}
observables = {}
pi = []
transitions = []
emissions = []


def load_hmm(afile):
  global transitions, hidden, observables, pi, emissions

  f = open(afile, 'r')
  
  # skip three commentary lines + newline
  f.readline(); f.readline(); f.readline(); f.readline()
  
  
  f.readline() # skip line 'hidden'
  hidden = {value: key for (key, value) in \
    enumerate(str.split(f.readline().rstrip('\n'), ' '))}
  
  f.readline(); f.readline() # skip lines 'observables' + newline
  observables = {value: key for (key, value) in \
    enumerate(str.split(f.readline().rstrip('\n'), ' '))}
  
  f.readline(); f.readline() # skip lines 'pi' + newline
  pi = [float(fl) for fl in str.split(f.readline().rstrip('\n'), ' ')]

  f.readline(); f.readline(); # skip lines 'transitions' + newline
  curr_line = f.readline()
  
  while not curr_line == '\n':
    transitions.append(np.array([float(fl) for fl in str.split(curr_line, ' ')]))
    curr_line = f.readline()
  transitions = np.array(transitions)

  f.readline(); # skip line 'emissions'
  curr_line = f.readline();
  while not curr_line == '\n':
    emissions.append(np.array([float(fl) for fl in str.split(curr_line, ' ')]))
    curr_line = f.readline()
  emissions = np.array(emissions)

def write_hmm(outputfile):
  output = open(outputfile, 'w')
  # write our model at the top of the output file
  output.write(str(sorted(hidden.items(), key=lambda x: x[1]))+'\n')
  output.write(str(transitions)+'\n')
  output.write(str(pi)+'\n')
  output.write(str(sorted(observables.items(), key=lambda x: x[1]))+'\n')
  output.write(str(emissions)+'\n')
  output.close()

def process_sequencefile(argfile, outputfile, hidden_included=False):
  # write_hmm(outputfile)

  f = open(argfile, 'r')
  curr_line = f.readline()

  with open(outputfile, 'w') as output:
    # stops when whole sequence file has been processed 
    #  (reaches an empty line where a name was expected)
    while curr_line.strip():
      name = curr_line[1:].rstrip()
      observed_seq = f.readline().strip()
      if (hidden_included):
        hidden_seq = f.readline()[2:].strip()

      # WRITE LOG JOINT PROBABILITY TO OUTPUT FILE
      # log_joint_prob = log_joint_prob(observed_seq, hidden_seq)
      # output.write("%s\nlog P(x,z) = %f\n\n" % (name, log_joint_prob))

      # WRITE VITERBI HIDDEN SEQUENCE PREDICTION AND PROBABILITY OF SEQUENCE
    #   log_most_likely_prob, hidden_seq = viterbi_logspace_backtrack(observed_seq)
      log_most_likely_prob, hidden_seq = posterior_sequence_decoding(observed_seq)
      str_viterbi_result = ""
      str_viterbi_result += '>'+name+'\n'
      str_viterbi_result += observed_seq+'\n'
      str_viterbi_result += '#\n'
      str_viterbi_result += hidden_seq+'\n'
      str_viterbi_result += '; log P(x,z) = %f\n\n' % log_most_likely_prob
      output.write(str_viterbi_result)
      
      f.readline() # skip empty separation line
      curr_line = f.readline() # read next sequence name, empty if at EOF

  f.close()


def log_joint_prob(observed_seq, hidden_seq):
  # probability of initialising in first hidden state
  init_state_prob = np.log(pi[hidden[hidden_seq[0]]])
  log_joint_prob = init_state_prob

  for i, c in enumerate(observed_seq):
    curr_hidden = hidden[hidden_seq[i]]
    curr_obs = observables[observed_seq[i]]

    # emitting current symbol in current state
    curr_emission_prob = np.log(emissions[curr_hidden][curr_obs])
    log_joint_prob += curr_emission_prob

    # transitioning to next state, if any
    if i < len(observed_seq)-1:
      next_hidden = hidden[hidden_seq[i+1]]
      curr_transition_prob = np.log(transitions[curr_hidden][next_hidden])
      log_joint_prob += curr_transition_prob

  return log_joint_prob

def viterbi_logspace_backtrack(aSequence):
  # BASIS
  idx_first_obs = observables.get(aSequence[0])
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
    idx_curr_obs = observables.get(aSequence[obs])
    
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
      idx_obs = observables.get(x[n+1])

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

  # ugly conversion from 'z' (hidden state number) to string of actual hidden 
  # state names
  hidden_seq = ""
  for i in z:
    for (k, v) in hidden.items():
      if v == i-1:
        hidden_seq += k
  
  return (log_most_likely_prob, hidden_seq)

def logsum(log_x, log_y):
  if (log_x == float('-inf')):
    return log_y
  if (log_y == float('-inf')):
    return log_x

  if (log_x > log_y):
    return log_x + np.log(1 + 2**(log_y - log_x))
  else:
    return log_y + np.log(1 + 2**(log_x - log_y))

def forward(aSequence):
  # BASIS
  idx_first_obs = observables.get(aSequence[0])
  alpha = np.log([pi]) + np.log(emissions[:,idx_first_obs])
  
  # RECURSIVE
  for obs in range(1, len(aSequence)):

    logsum_vector = []

    # iterating through all states to generate next col in alpha
    for i, _ in enumerate(hidden):

      # find transition probabilities from every state to this current state
      trans_to_state_i = np.log(transitions[:,i])
      
      # fetch previous col in alpha
      prev_alpha_col = alpha[-1]

      curr_hidden_log_sum = float('-inf')
      for j, _ in enumerate(hidden):

        # find the 'logspace' sum of probabilities that this state will follow
        # from the prev col
        curr_hidden_log_sum = logsum(curr_hidden_log_sum, \
                                     prev_alpha_col[j] + trans_to_state_i[j])

      # save for multiplying with emission probabilities to determine alpha col
      logsum_vector.append(curr_hidden_log_sum)

    # get idx of current observation to use with defined matrix data structures
    idx_curr_obs = observables.get(aSequence[obs])
    
    # get emission probabilities of current observation for all states
    emissions_curr_obs = emissions[:,idx_curr_obs]
    
    # create and add the new col to the alpha table
    new_alpha_col = np.log(emissions_curr_obs) + logsum_vector

    alpha = np.append(alpha, [new_alpha_col], axis=0)
  
  return alpha

def scaling_forward(aSequence):
  c = []
  
  # BASIS
  idx_first_obs = observables.get(aSequence[0])
  c1 = np.sum(pi * emissions[:,idx_first_obs])
  c.append(c1)

  alpha_z1 = pi * emissions[:,idx_first_obs]

  alpha_hat = []
  alpha_hat.append(alpha_z1/c1)
  
  # basis down, we've now got a^(z1) and c1

  # RECURSION
  delta = []
  for obs in range(1, len(aSequence)):

    # get idx of current observation to use with defined matrix data structures
    idx_curr_obs = observables.get(aSequence[obs])
    
    # get emission probabilities of current observation for all states
    emissions_curr_obs = emissions[:,idx_curr_obs]

    prev_alpha_hat_col = alpha_hat[-1]

    nth_delta_vector = []

    # iterating through all states to generate next col in delta
    for i, _ in enumerate(hidden):
      # find transition probabilities from every state to this current state
      trans_to_state_i = transitions[:,i]
      nth_delta_vector.append(np.dot(trans_to_state_i, prev_alpha_hat_col))

    nth_delta_vector = emissions_curr_obs * nth_delta_vector
    
    delta.append(nth_delta_vector)

    # NOW:  compute and store c_n as:
    #          sum from 1 to K as k: delta(z_nk)
    
    c_n = np.sum(nth_delta_vector)
    c.append(c_n)
    
    # THEN: compute and store a^(z_nk) as:
    #          delta(z_nk) / c_n
    
    alphahat_znk = nth_delta_vector / c_n
    
    alpha_hat.append(alphahat_znk)
  
  return c, np.array(alpha_hat)
  
def scaling_backward(aSequence, c):
  # BASIS
  beta_hat = []
  
  # b^(z_N) = 1 for all k possible z_N
  beta_hat.append(np.ones(len(hidden)))
  
  
  # RECURSION
  epsilon = []
  for obs in range(len(aSequence)-2, -1, -1):
    
    # get idx of next observation (n+1) to use with defined matrix data structures
    idx_curr_obs = observables.get(aSequence[obs+1])
    
    # beta_hat[-1]
    
    # get emission probabilities of current observation for all states
    emissions_curr_obs = emissions[:,idx_curr_obs]
    
    nth_epsilon_vector = []
    
    for i, _ in enumerate(hidden):
      # transition from state 'i' to all k hidden states
      trans_from_state_i = transitions[i,:]
      
      temp_prob_prod = emissions_curr_obs * trans_from_state_i
      nth_epsilon_vector.append(np.dot(temp_prob_prod, beta_hat[-1]))
    
    
    epsilon.append(nth_epsilon_vector)
    
    betahat_znk = nth_epsilon_vector / c[obs+1]
    
    beta_hat.append(betahat_znk)
    
  beta_hat.reverse()
  return np.array(beta_hat)

def posterior(aSequence, n):
  c, alpha_hat = scaling_forward(aSequence)
  beta_hat = scaling_backward(aSequence, c)
  
  curr_alphahat_vect = alpha_hat[n]
  curr_betahat_vect = beta_hat[n]
    
  vect_prod = curr_alphahat_vect * curr_betahat_vect
  idx_most_likely_state = np.argmax(vect_prod)
  
  # ugly conversion from 'z' (hidden state number) to string of actual hidden 
  # state names
  for (k, v) in hidden.items():
        if v == idx_most_likely_state:
          return k
  return None
    

def posterior_sequence_decoding(aSequence):
  hidden_seq = ""
  for obs in range(0, len(aSequence)):
    hidden_seq += posterior(aSequence, obs)
  joint_prob = log_joint_prob(aSequence, hidden_seq)
  
  return joint_prob, hidden_seq
 
 
def scale(unscaled_value, scalar):
  # going from entries in 'alpha' to entries in 'alpha_hat' using the 
  # corresponding scalar from 'c'
  return np.e**unscaled_value / scalar


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("hmm", help="Hidden Markov Model specification file")
  parser.add_argument("sequences", help="File containing input sequences")
  parser.add_argument("-o", "--output", type=str, help="File to output results to")
  args = parser.parse_args()
  
  hmm_file = args.hmm
  sequences_file = args.sequences
  output_file = "output.txt"

  if (args.output is not None):
    output_file = args.output

  load_hmm(hmm_file)
  
  process_sequencefile(sequences_file, output_file)
  
#   observed = "MAKNLILWLVIAVVLMSVFQSFGPSESNGRKVDYSTFLQEVNNDQVREARINGREINVTKKDSNRYTTYIPVQDPKLLDNLLTKNVKVVGEPPEEPSLLASIFISWFPMLLLIGVWIFFMRQMQGGGGKGAMSFGKSKARMLTEDQIKTTFADVAGCDEAKEEVAELVEYLREPSRFQKLGGKIPKGVLMVGPPGTGKTLLAKAIAGEAKVPFFTISGSDFVEMFVGVGASRVRDMFEQAKKAAPCIIFIDEIDAVGRQRGAGLGGGHDEREQTLNQMLVEMDGFEGNEGIIVIAATNRPDVLDPALLRPGRFDRQVVVGLPDVRGREQILKVHMRRVPLAPDIDAAIIARGTPGFSGADLANLVNEAALFAARGNKRVVSMVEFEKAKDKIMMGAERRSMVMTEAQKESTAYHEAGHAIIGRLVPEHDPVHKVTIIPRGRALGVTFFLPEGDAISASRQKLESQISTLYGGRLAEEIIYGPEHVSTGASNDIKVATNLARNMVTQWGFSEKLGPLLYAEEEGEVFLGRSVAKAKHMSDETARIIDQEVKALIERNYNRARQLLTDNMDILHAMKDALMKYETIDAPQIDDLMARRDVRPPAGWEEPGASNNSGDNGSPKAPRPVDEPRTPNPGNTMSEQLGDK"
#   hidden_seq = posterior_sequence_decoding(observed)
#   print(log_joint_prob(observed, hidden_seq))


if __name__ == '__main__':
  main()
  
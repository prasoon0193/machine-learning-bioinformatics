import numpy as np

hidden = {}
observables = {}
pi = []
transitions = []
emissions = []

def hmm2latex(aFile):
  global hidden, observables, pi, transitions, emissions

  hidden = {}
  observables = {}
  pi = []
  transitions = []
  emissions = []

  with open(aFile, 'r') as f:
    # skip three commentary lines + newline
    f.readline(); f.readline(); f.readline(); f.readline()
    
    
    f.readline() # skip line 'hidden'
    hidden = {value: key for (key, value) in \
      enumerate(str.split(f.readline().rstrip('\n'), ' '))}
    
    f.readline(); f.readline() # skip lines 'observables' + newline
    observables = {value: key for (key, value) in \
      enumerate(str.split(f.readline().rstrip('\n'), ' '))}
    
    f.readline(); f.readline() # skip lines 'pi' + newline
    pi = [float(fl) for fl in str.split(f.readline().rstrip(), ' ')]
  
    f.readline(); f.readline(); # skip lines 'transitions' + newline
    curr_line = f.readline()
    
    while not curr_line in ['\n', '']:
      transitions.append(np.array([float(fl) for fl in str.split(curr_line, ' ')]))
      curr_line = f.readline().rstrip()
    transitions = np.array(transitions)
  
    f.readline(); # skip line 'emissions'
    curr_line = f.readline().rstrip();
    while not curr_line in ['\n', '']:
      emissions.append(np.array([float(fl) for fl in str.split(curr_line, ' ')]))
      curr_line = f.readline().rstrip()
    emissions = np.array(emissions)

  with open('output.txt', 'w') as output:
    head = '\\documentclass{standalone}\n'+ \
      '\\usepackage{units}\n'+ \
      '\\usepackage{ifthen}\n'+ \
      '\\usepackage{tikz}\n'+ \
      '\\usetikzlibrary{calc}\n'+ \
      '\n'+ \
      '\\begin{document}\n'+ \
      '\\tikzstyle{vertex}=[draw,black,fill=blue,circle,minimum size=10pt,inner sep=0pt]\n'+ \
      '\\tikzstyle{edge}=[very thick]\n'+ \
      '\\begin{tikzpicture}\n'
    output.write(head)

    nodes = ''
    for i, (k, v) in enumerate(sorted(hidden.items(), key=lambda x: x[1])):
      nodes += '\\node ('+k+')[vertex,fill=gray!10,align=left] at ('+str(v*10)+',0) \n'

      # loop through and add emissions for this node
      nodes += '{'
      for obs, iobs in sorted(observables.items(), key=lambda x: x[1]):
        obs_emitprob = emissions[v,iobs]
        nodes += '$'+obs+'~'+str(("%.5f" % obs_emitprob))+'$\\\\'
      nodes += '};\n\n'
    output.write(nodes)

    edges = ''
    for i, (k, v) in enumerate(sorted(hidden.items(), key=lambda x: x[1])):
      for j, (k2, v2) in enumerate(sorted(hidden.items(), key=lambda x: x[1])):
        if transitions[v, v2] == 0:
          # skip transitions with no probability of happening
          continue;
        mode = '[loop above, looseness=5]' if k == k2 else '[bend left, looseness=1]'
        trans_prob = str(("%.5f" % transitions[v, v2]))
        edges += '\\path[thick,->] ('+k+')    edge '+mode+'node [anchor=center,above,sloped] {$'+trans_prob+'$} ('+k2+');\n'
    output.write(edges)


    foot = '\\end{tikzpicture}\n\\end{document}\n'
    output.write(foot)

    


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
  hmm2latex('helix-hmm.txt')
  print_hmm()

if __name__ == '__main__':
  main()

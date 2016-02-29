#
# compare_tm_pred.py <true> <pred>
#
# Compares a predicted trans-membrane structure against the true trans-membrane structure
# and computes various statistics summarizing the quality of the prediction. The comparison
# only focuses on the location of the membranes.
#
# The files <true> and <pred> are the true and the predicted structures respectively. The
# files can contain several structures cf. format used in the projects in MLiB Q3/2015.
#
# Christian Storm Pedersen, 07-feb-2015


import sys
import string
import math

#from fasta import fasta

def fasta(f):
    """
    Reads the fasta file f and returns a dictionary with the sequence names as keys and the
    sequences as the corresponding values. Lines starting with ';' in the fasta file are
    considered comments and ignored.
    """
    d = {}
    curr_key = ""
    lines = [string.strip(l) for l in open(f).readlines() if (l[0] != ';')]
    for l in lines:
        if l == '': continue
        if l[0] == '>':
            if curr_key != "": d[curr_key] = curr_val
            curr_key = l[1:]
            curr_val = ""
        else:
            curr_val = curr_val + l
    d[curr_key] = curr_val
    
    return d

def count(true, pred):
    tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'M':
            if true[i] == 'M':
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if true[i] == 'i' or true[i] == 'o':
                tn = tn + 1
            else:
                fn = fn + 1

    return tp, fp, tn, fn

def print_stats(tp, fp, tn, fn):
    sn = sp = cc = acp = float('Inf')
    try:
        sn = float(tp) / (tp + fn)
        sp = float(tp) / (tp + fp)
        cc = float((tp*tn - fp*fn)) / math.sqrt(float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
        acp = 0.25 * (float(tp)/(tp+fn) + float(tp)/(tp+fp) + float(tn)/(tn+fp) + float(tn)/(tn+fn))
    except ZeroDivisionError:
        None
    ac = (acp - 0.5) * 2
    print("Sn = %.4f, Sp = %.4f, CC = %.4f, AC = %.4f" % (sn, sp, cc, ac))

true = fasta(sys.argv[1])
pred = fasta(sys.argv[2])

total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

for key in sorted(true.keys()):
    true_x, true_z = [string.strip(s) for s in true[key].split('#')]
    pred_x, pred_z = [string.strip(s) for s in pred[key].split('#')]

    if len(pred_x) != len(pred_z):
        print "ERROR: prediction on %s has wrong length" % (key)
        sys.exit(1)

    print ">" + key
    tp, fp, tn, fn = count(true_z, pred_z)
    total_tp, total_fp, total_tn, total_fn = total_tp + tp, total_fp + fp, total_tn + tn, total_fn + fn
    print_stats(tp, fp, tn, fn)
    print

print "Summary (over all sequences):"
print_stats(total_tp, total_fp, total_tn, total_fn)

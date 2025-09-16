#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import math

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

co_occur = defaultdict(set) 
fe_count = defaultdict(int)
for (fwords, ewords) in bitext:
    for f in fwords:
        for e in ewords:
            co_occur[e].add(f)
            fe_count[(f,e)] += 1


t = defaultdict(lambda: defaultdict(float))  # t[e][f] = P(e | f)
Fs = set()
Es = set()
for (fwords, ewords) in bitext:
    Fs.update(fwords)
    Es.update(ewords)
f_to_es = defaultdict(set)
for (f,e) in fe_count.keys():
    f_to_es[f].add(e)
for f in Fs:
    es = list(f_to_es[f]) if f in f_to_es else list(Es)
    if len(es) == 0:
        es = list(Es)
    p = 1.0 / len(es)
    for e in es:
        t[e][f] = p

s = defaultdict(float) 
INIT_WINDOW = 5
for d in range(-INIT_WINDOW, INIT_WINDOW+1):
    s[d] = 1.0
def normalize_s(sdict):
    total = sum(sdict.values())
    if total == 0:
        return
    for k in list(sdict.keys()):
        sdict[k] /= total

normalize_s(s)

def compute_trans_probs_for_sentence(le):
    positions = list(range(1, le+1))
    # if opts.use_null:
    #     positions = [0] + positions
    trans = {}
    for i_prev in positions:
        trans[i_prev] = {}
        total = 0.0
        for i in positions:
            delta = i - i_prev
            trans[i_prev][i] = s.get(delta, 1e-12)
            total += trans[i_prev][i]
        if total == 0:
            for i in trans[i_prev]:
                trans[i_prev][i] = 1.0 / len(trans[i_prev])
        else:
            for i in trans[i_prev]:
                trans[i_prev][i] /= total
    return trans

def viterbi_align(fwords, ewords):
    J = len(fwords)
    L = len(ewords)
    positions = list(range(1, L+1))
    # if opts.use_null:
    #     positions = [0] + positions

    trans = compute_trans_probs_for_sentence(L)

    NEG_INF = -1e9
    def log_t(e_word, f_word):
        v = t.get(e_word, {}).get(f_word, 0.0)
        if v <= 0:
            return math.log(1e-12)
        return math.log(v)

    dp = [defaultdict(lambda: NEG_INF) for _ in range(J+1)]
    backptr = [dict() for _ in range(J+1)]

    for i in positions:
        start_log = math.log(1.0 / len(positions))
        e_word = ewords[i-1] if i != 0 else "NULL"
        dp[1][i] = start_log + log_t(e_word, fwords[0])
        backptr[1][i] = None

    for j in range(2, J+1):
        fj = fwords[j-1]
        for i in positions:
            best_score = NEG_INF
            best_prev = None
            e_word = ewords[i-1] if i != 0 else "NULL"
            emis = log_t(e_word, fj)
            for i_prev in positions:
                trans_prob = trans[i_prev].get(i, 1e-12)
                if trans_prob <= 0:
                    score = NEG_INF
                else:
                    score = dp[j-1][i_prev] + math.log(trans_prob) + emis
                if score > best_score:
                    best_score = score
                    best_prev = i_prev
            dp[j][i] = best_score
            backptr[j][i] = best_prev

    best_i = None
    best_score = NEG_INF
    for i in positions:
        if dp[J][i] > best_score:
            best_score = dp[J][i]
            best_i = i

    a = [None] * (J + 1) 
    cur = best_i
    j = J
    while j >= 1:
        a[j] = cur
        cur = backptr[j][cur]
        j -= 1
    return a[1:]

iteration=5
for it in range(iteration):
    count_fe = defaultdict(float)      # count of (f_j aligned to e_i): key (f_word, e_word)
    total_f = defaultdict(float)       # for normalization per f_word
    count_jump = defaultdict(float)    # counts for delta = i - i_prev

    for (idx, (fwords, ewords)) in enumerate(bitext):
        if len(fwords) == 0 or len(ewords) == 0:
            continue
        # run viterbi to get a_j for each j
        a = viterbi_align(fwords, ewords)  # length J, entries in positions (0..L)
        # accumulate emission counts
        for j, i in enumerate(a):
            fj = fwords[j]
            if i == 0:  # NULL alignment, we skip emission accumulation to keep same output format (don't align to NULL)
                # but you might still want to count NULL emissions (optional)
                continue
            ej = ewords[i-1]
            count_fe[(fj, ej)] += 1.0
            total_f[fj] += 1.0
        # accumulate transition (j-1 -> j) jumps
        prev = None
        for j, i in enumerate(a, start=1):
            if prev is None:
                # treat start as previous position 0 if opts.use_null False we still use 0 as start
                prev = 0
            delta = i - prev
            count_jump[delta] += 1.0
            prev = i

        # if idx % 2000 == 0 and idx > 0:
        #     sys.stderr.write(".")

    # M-step: update t and s
    # Update t[e][f] = count(f->e) / total_f[f]
    # If a f has zero total (rare), keep previous distribution
    # Clear and reassign t to avoid stale entries
    new_t = defaultdict(lambda: defaultdict(float))
    for (f_e, cnt) in count_fe.items():
        fj, ej = f_e
        new_t[ej][fj] = cnt
    # normalize by total_f per f
    for fj in list(total_f.keys()):
        denom = total_f[fj]
        if denom <= 0:
            continue
        # find all ej with counts for this fj
        for ej in list(new_t.keys()):
            if new_t[ej].get(fj, 0.0) > 0:
                new_t[ej][fj] /= denom
    # For any (ej,fj) pair missed (e.g., previously present but zero now), keep a small smoothing
    # Add tiny uniform smoothing over e for stability
    for fj in Fs:
        # ensure at least one e has nonzero prob for this f
        sum_p = 0.0
        for ej in Es:
            sum_p += new_t[ej].get(fj, 0.0)
        if sum_p == 0.0:
            # fallback to previous t distribution (or uniform)
            es = list(f_to_es.get(fj, Es))
            p = 1.0 / len(es)
            for ej in es:
                new_t[ej][fj] = p
        else:
            # renormalize across ej for this fj
            for ej in Es:
                if new_t[ej].get(fj, 0.0) > 0:
                    # already normalized
                    pass
    t = new_t

    # Update s (jump distribution)
    if len(count_jump) == 0:
        # fallback keep s
        pass
    else:
        s = dict(count_jump)
        normalize_s(s)



# Output alignments in same format as original align.py: for each sentence print fIndex-eIndex pairs
for (fwords, ewords) in bitext:
    if len(fwords) == 0 or len(ewords) == 0:
        print()
        continue
    a = viterbi_align(fwords, ewords)
    out_pairs = []
    for j, i in enumerate(a):
        # only output if aligned to a real english position (not NULL)
        if i == 0:
            continue
        # out_pairs.append("%d-%d" % (j, i-1))  # j is f-index (0-based), i-1 is e-index (0-based)
        sys.stdout.write("%i-%i " % (j, i-1))
    # print(" ".join(out_pairs))
    sys.stdout.write("\n")
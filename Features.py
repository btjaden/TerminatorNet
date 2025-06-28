
import warnings; warnings.filterwarnings("ignore")
import sys, random, math
import numpy as np
import scipy.stats as stats
import multiprocessing
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from Energy import *


###############################
#####   ENERGY FEATURES   #####
###############################

MAX_LOOP_ENERGY = 6.2
U_TRACT_LENGTH = 8
B1, B4, Beta1, Beta4 = 0.005, 6.0, 0.6, 0.45
U_INIT = 3.1  # Taken from Sugimoto 1995
U_TRACT = {'AA':-1.0,'AC':-2.1,'AG':-1.8,'AT':-0.9,'CA':-0.9,'CC':-2.1,'CG':-1.7,'CT':-0.9,
           'GA':-1.3,'GC':-2.7,'GG':-2.9,'GT':-1.1,'TA':-0.6,'TC':-1.5,'TG':-1.6,'TT':-0.2}
STACK = {'ATAT':-0.9,'ATCG':-2.2,'ATGC':-2.1,'ATGT':-0.6,'ATTA':-1.1,'ATTG':-1.4,
         'CGAT':-2.1,'CGCG':-3.3,'CGGC':-2.4,'CGGT':-1.4,'CGTA':-2.1,'CGTG':-2.1,
         'GCAT':-2.4,'GTAT':-1.3,'GCCG':-3.4,'GTCG':-2.5,'GCGC':-3.3,'GCGT':-1.5,
         'GTGC':-2.1,'GTGT':-0.5,'GCTA':-2.2,'GCTG':-2.5,'GTTA':-1.4,'GTTG':1.3,
         'TAAT':-1.3,'TGAT':-1.0,'TACG':-2.4,'TGCG':-1.5,'TAGC':-2.1,'TAGT':-1.0,
         'TGGC':-1.4,'TGGT':0.3,'TATA':-0.9,'TATG':-1.3,'TGTA':-0.6,'TGTG':-0.5}

# Energy of U-tract. Input is U-tract sequence.
def G_U(s):
    G = U_INIT
    for i in range(min(U_TRACT_LENGTH-1, len(s)-1)): G += U_TRACT[s[i:i+2]]
    return G


# Energy of stem base. Input is hairpin sequence.
def G_B(s):
    G = 0.0
    no_basepair = False
    for i in range(3):  # Use 3 stacking pairs at base of stem
        pair = s[i] + s[-(i+1)] + s[i+1] + s[-(i+2)]
        if (pair in STACK): G += STACK[pair]
    return G


# Energy of structure.
def G_H(s, calculate_structure=False):
    G, structure = compute_structure_with_minimum_energy(s, calculate_structure, False, 7)
    if (calculate_structure): return G, structure
    else: return G


# Energy of loop. Includes closing basepair.
def G_L(s):
    if (len(s) == 0): return MAX_LOOP_ENERGY
    G = hairpin(s, 0, len(s)-1)
    return min(G, MAX_LOOP_ENERGY)


# Energy of A-tract. Input is A-tract, first nt of stem, final nt of stem, U-tract.
def G_A(s1, char1, char2, s2):
    G, _1, _2, _h = compute_hybridization_with_minimum_energy(s1 + char1, char2 + s2, False)
    return G + 1.5  # Error adjustment


# Terminator strength
def T_S(GU, GL, GA, GB):
    denominator = B1*math.exp(Beta1*GL) + B4*math.exp(Beta4*(GB+GA-GU))*(1+B1*math.exp(Beta1*GL))
    return 1.0 + 1.0/denominator


###########################
#####   TT FEATURES   #####
###########################

INF = 99.9
MAX_HAIRPIN, MIN_HAIRPIN, MAX_LOOP, MIN_LOOP, MIN_STEM = 59, 11, 13, 3, 4
GC, AT, GT, MISMATCH, GAP = -2.3, -0.9, 1.3, 3.5, 6.0
ENERGY = {'GC':GC, 'CG':GC, 'AT':AT, 'TA':AT, 'GT':GT, 'TG':GT, 'AA':MISMATCH,
          'AC':MISMATCH, 'AG':MISMATCH, 'CA':MISMATCH, 'CC':MISMATCH, 'CT':MISMATCH,
          'GA':MISMATCH, 'GG':MISMATCH, 'TC':MISMATCH, 'TT':MISMATCH}
def loop(n): return INF if (n > MAX_LOOP) else n - 2.0
TT_TAIL_LENGTH = 15
MAX_TAIL_ENERGY, MAX_HAIRPIN_ENERGY = -2.5, -2.0
RANDOM_SEQUENCE_LENGTH = 20000000
MAX_SEQ_LENGTH = 100
T = np.full((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH), INF)  # Energy table
B = np.full((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH), -1)   # Backtrack table
GC_PARAMS = {0.20:[267665, 1.6770755130419306, -1.063561975106122, 1.2896782985770128, 5.280503573764744, 2.5272786697943173, 0.3677746767992657],
             0.25:[142384, 1.6354597236353605, -1.0407764690616221, 1.263273128374113, 6.301846901623428, 2.4015388195266234, 0.34619090082559467],
             0.30:[63331, 1.7730719951841698, -1.074924460910759, 1.2313055334406298, 7.506960291877634, 2.2579491895063004, 0.3249814909635603],
             0.35:[37213, 2.314634352633771, -1.2987553286398237, 1.163816186437339, 7.13973367002874, 2.305700408556535, 0.33307307440529416],
             0.40:[68299, 2.5448969983827885, -1.3582493695725462, 1.1751310369372203, 5.186977804326571, 2.5404143973518734, 0.3707905839062232],
             0.45:[113092, 2.802495928252414, -1.4263826732916014, 1.1874911699888606, 3.8350566125126306, 2.6984652558346323, 0.4075782834891387],
             0.50:[171894, 3.1064666286277713, -1.513181245985507, 1.1962131706343566, 2.9272828674941493, 2.7910812691043168, 0.4408289318664924],
             0.55:[243029, 3.3784715302998265, -1.5836830014259458, 1.220507224628586,  2.2341204514538404, 2.8551511790418793, 0.47486636786015635],
             0.60:[322059, 3.6998031483849125, -1.6928682483581985, 1.2575505522449761, 1.6342762827438375, 2.9015510791948067, 0.5278123508967134],
             0.65:[399925, 4.029365916106801, -1.8029312747362654, 1.3066930829792183, 1.202738205163861,  2.9207256961906847, 0.5864294177706434],
             0.70:[457244, 4.219235176469958, -1.8697485696596288, 1.4177180418052147, 0.7894452771797953, 2.9234131256770555, 0.7781611352781375],
             0.75:[474156, 4.388762874428863, -1.9337859142428615, 1.5842841441399953, 0.6224400604174656, 2.9234131256770555, 0.569892306172348],
             0.80:[429709, 4.8032748759523, -2.1256971979736656, 1.7490484054686268, 0.4816914076146752, 2.9234131256770555, 0.48584932108397083]}


def backtrack(s, B, i, j):
    seq_pre, seq_post, struct_pre, struct_post = '', '', '', ''
    while (i < j):
        if (B[i, j] == 1):  # Pair
            char1, char2 = '(', ')'
            if (ENERGY[s[i] + s[j]] == MISMATCH): char1, char2 = '.', '.'
            seq_pre += s[i]
            seq_post += s[j]
            struct_pre += char1
            struct_post += char2
            i, j = i + 1, j - 1
        elif (B[i, j] == 0):  # Loop
            seq = seq_pre + ' ' + s[i:j+1] + ' ' + seq_post[::-1]
            struct = struct_pre + ' ' + ('.' * (j-i+1)) + ' ' + struct_post[::-1]
            j = i
            return seq, struct
        elif (B[i, j] == 2):  # Gap
            seq_pre += s[i]
            struct_pre += '.'
            i += 1
        elif (B[i, j] == 3):  # Gap
            seq_post += s[j]
            struct_post += '.'
            j -= 1
        else: sys.stderr.write('\nBacktracking error - unreachable case\n\n'); i=j
    return '', ''


# COMPUTES HAIRPIN_SCORE FOR SEQUENCE WHERE HAIRPIN STRUCTURE ENDS AT INDEX "END"
def hairpin_TT(s, end, perform_backtrack):
    #T = np.full((len(s), len(s)), INF)  # Energy table
    #B = np.full((len(s), len(s)), -1)   # Backtrack table
    s = s[max(end-MAX_HAIRPIN+1, 0):end+1]
    for i in range(len(s)-1, -1, -1):
        for j in range(i, len(s)):
            if ((j - i + 1) < MIN_LOOP): T[i, j]  = INF  # Loop too small
            elif ((j - i + 1) == MIN_LOOP): T[i, j], B[i, j] = loop(j - i + 1), 0
            else:
                _loop = loop(j - i + 1)
                _pair = ENERGY[s[i] + s[j]] + T[i+1, j-1]
                _gap1 = GAP + T[i+1, j]
                _gap2 = GAP + T[i, j-1]
                min_energy = min(_loop, _pair, _gap1, _gap2)
                T[i, j] = min_energy
                if (min_energy == _pair): B[i, j] = 1
                elif (min_energy == _loop): B[i, j] = 0
                elif (min_energy == _gap1): B[i, j] = 2
                elif (min_energy == _gap2): B[i ,j] = 3
    min_index = T.argmin(axis=0)[len(s)-1]
    if (perform_backtrack):
        seq, struct = backtrack(s, B, min_index, len(s)-1)
        return T[min_index, len(s)-1], seq, struct
    else: return T[min_index, len(s)-1]


# COMPUTES TAIL_SCORE FOR SEQUENCE WHERE TAIL STARTS AT INDEX "START"
def tail_TT(s, start):
    scores, x = [], 1.0
    for i in range(start, start+TT_TAIL_LENGTH):
        if (i < len(s)) and (s[i] == 'T'): x *= 0.9
        else: x *= 0.6
        scores.append(x)
    return 0 - sum(scores)


# GENERATES CONFIDENCE TABLES FROM RANDOM SEQUENCES (ONLY NEEDS TO BE DONE ONCE!)
def confidence_tables():
    out_file = open('gamma_values.txt', 'w')
    for GC in np.arange(0.2, 0.85, 0.05):
        sys.stderr.write('GC content:\t' + str(round(GC, 2)) + '\n')
        s = random_sequence(RANDOM_SEQUENCE_LENGTH, GC)
        hairpins, tails = [], []
        for k in range(min(MIN_HAIRPIN, len(s)-2), len(s)-1):
            if (s[k-1] != 'T') and (s[k:k+2] == 'TT'):  # Only search when two TT are found
                current_energy, current_seq, current_struct = hairpin(s, k-1)
                parse_seq = current_seq.split()
                if (len(parse_seq[0]) < MIN_STEM) or (len(parse_seq[-1]) < MIN_STEM): continue
                tail_energy = tail(s, k)
                hairpins.append(current_energy)
                tails.append(tail_energy)
        hairpins = -np.array(hairpins)
        tails = -np.array(tails)
        alpha_H, loc_H, beta_H = stats.gamma.fit(hairpins)
        alpha_T, loc_T, beta_T = stats.gamma.fit(tails)
        out_file.write(str(round(GC, 2)) + '\t' + str(len(hairpins)) + '\t' + str(alpha_H) + '\t' + str(loc_H) + '\t' + str(beta_H) + '\t' + str(alpha_T) + '\t' + str(loc_T) + '\t' + str(beta_T) + '\n')
    out_file.close()


# CONFIDENCE SCORE OF TERMINATOR (BETWEEN 0 AND 100)
def score(GC, hairpin_energy, tail_energy):
    if (GC < 0.2): GC_bin = 0.2  # Ensure GC-content is one of 0.20, 0.25, 0.30, ..., 0.75, 0.80
    elif (GC > 0.8): GC_bin = 0.8
    else: GC_bin = 5 * round(100.0*GC / 5.0) / 100.0
    count, alpha_H, loc_H, beta_H, alpha_T, loc_T, beta_T = GC_PARAMS[GC_bin]
    cdf_H = stats.gamma.cdf(-hairpin_energy, a=alpha_H, loc=loc_H, scale=beta_H)
    cdf_T = stats.gamma.cdf(-tail_energy, a=alpha_T, loc=loc_T, scale=beta_T)
    examples = max(1, round(count * (1.0-cdf_H) * (1.0-cdf_T)))
    return round((-100.0 / math.log(count)) * math.log(examples / float(count)))


#####################
#####   UTILS   #####
#####################

# GENERATES A RANDOM SEQUENCE OF THE GIVEN LENGTH WITH THE SPECIFIED GC CONTENT
def random_sequence(length, gc=0.5):
    s = []
    for i in range(length):
        r = random.random()
        if (r < gc/2.0): s.append('C')
        elif (r < gc): s.append('G')
        elif (r < gc+(1.0-gc)/2.0): s.append('A')
        else: s.append('T')
    return ''.join(s)


def reverse_complement(s):
    s = s.replace('A', '?')
    s = s.replace('T', 'A')
    s = s.replace('?', 'T')
    s = s.replace('C', '?')
    s = s.replace('G', 'C')
    s = s.replace('?', 'G')
    return s[::-1]


#################################
#####   HELPFUL FUNCTIONS   #####
#################################

def get_structure_components(structure):
    hairpin_start = structure.find('(')
    if (hairpin_start == -1): return -1, -1, -1, -1, 0, 0, 0, len(structure), 0, 0, 0, 0
    hairpin_end = structure.rfind(')')
    loop_end = structure.find(')')
    loop_start = loop_end - 1
    while (structure[loop_start] != '('): loop_start -= 1
    hairpin = structure[hairpin_start:hairpin_end+1]
    hairpin_length = len(hairpin)
    loop_length = loop_end - loop_start + 1  # Includes closing BPs
    total_stacking = hairpin.count('(')
    total_unpaired = hairpin.count('.') - (loop_length - 2)
    base_stacking = 0
    while (hairpin[base_stacking] == '(') and (hairpin[-(base_stacking+1)] == ')'): base_stacking += 1
    num_loops1, num_loops2, consecutive_stacking1, consecutive_stacking2, max_loop = 0, 0, 0, 0, 0
    currently_in_loop, current_consecutive, current_loop_size = False, 0, 0
    for i in range(hairpin_start, loop_start+1):
        if (structure[i] == '('):  # In stacking
            current_consecutive += 1
            if (current_consecutive > consecutive_stacking1):
                consecutive_stacking1 = current_consecutive
            currently_in_loop = False
            current_loop_size = 0
        else:  # In loop
            if (not currently_in_loop):  # Starting a new loop
                num_loops1 += 1
                currently_in_loop = True
                current_consecutive = 0
            current_loop_size += 1
            if (current_loop_size > max_loop): max_loop = current_loop_size
    currently_in_loop, current_consecutive = False, 0
    for i in range(loop_end, hairpin_end+1):
        if (structure[i] == ')'):  # In stacking
            current_consecutive += 1
            if (current_consecutive > consecutive_stacking2):
                consecutive_stacking2 = current_consecutive
            currently_in_loop = False
            current_loop_size = 0
        else:  # In loop
            if (not currently_in_loop):  # Starting a new loop
                num_loops2 += 1
                currently_in_loop = True
                current_consecutive = 0
            current_loop_size += 1
            if (current_loop_size > max_loop): max_loop = current_loop_size
    num_loops = max(num_loops1, num_loops2)
    consecutive_stacking = min(consecutive_stacking1, consecutive_stacking2)
    return hairpin_start, hairpin_end, loop_start, loop_end, hairpin_length, loop_length, total_stacking, total_unpaired, base_stacking, num_loops, consecutive_stacking, max_loop


def get_structure_features(organism, hairpin, A_tract, U_tract):
    # TT features
    GC_CONTENT = {'Bacillus subtilis':0.44, 'Escherichia coli':0.51}
    TT_hairpin, hairpin_current, structure_TT = hairpin_TT(hairpin, len(hairpin)-1, True)
    hairpin_current = hairpin_current.replace(' ', '')
    structure_TT = structure_TT.replace(' ', '')
    TT_tail = tail_TT(U_tract, 0)
    TT_score = score(GC_CONTENT[organism], TT_hairpin, TT_tail)
    hairpin_start_TT, hairpin_end_TT, loop_start_TT, loop_end_TT, hairpin_length_TT, loop_length_TT, total_stacking_TT, total_unpaired_TT, base_stacking_TT, num_loops_TT, consecutive_stacking_TT, max_loop_TT = get_structure_components(structure_TT)
    TT_hairpin_avg = TT_hairpin / hairpin_length_TT if (hairpin_length_TT > 0) else 0.0

    # Energy features
    GU = G_U(U_tract)
    GB = G_B(hairpin)
    GA = G_A(A_tract, hairpin[0], hairpin[-1], U_tract)
    GH, structure = G_H(hairpin, True)
    hairpin_start, hairpin_end, loop_start, loop_end, hairpin_length, loop_length, total_stacking, total_unpaired, base_stacking, num_loops, consecutive_stacking, max_loop = get_structure_components(structure)
    GH_avg = GH / hairpin_length if (hairpin_length > 0) else 0.0
    GL = G_L(hairpin[loop_start:loop_end+1])
    TS = T_S(GU, GL, GA, GB)

    return TT_hairpin, TT_hairpin_avg, TT_tail, TT_score, structure_TT, hairpin_length_TT, loop_length_TT, total_stacking_TT, total_unpaired_TT, base_stacking_TT, num_loops_TT, consecutive_stacking_TT, max_loop_TT, hairpin_length, loop_length, total_stacking, total_unpaired, base_stacking, num_loops, consecutive_stacking, max_loop, GU, GB, GA, GH, GH_avg, GL, TS, structure


#############################
#####   SCAN SEQUENCE   #####
#############################

HAIRPIN_LENGTH, TAIL_LENGTH = 52, 8
WINDOW, STEP = HAIRPIN_LENGTH + TAIL_LENGTH, 1
FEATURES = ['TT_hairpin', 'TT_hairpin_avg', 'TT_tail', 'TT_score', 'hairpin_length_TT', 'loop_length_TT', 'total_stacking_TT', 'total_unpaired_TT', 'base_stacking_TT', 'num_loops_TT', 'consecutive_stacking_TT', 'max_loop_TT', 'hairpin_length', 'loop_length', 'total_stacking', 'total_unpaired', 'base_stacking', 'num_loops', 'consecutive_stacking', 'max_loop', 'GU', 'GB', 'GA', 'GH', 'GH_avg', 'GL', 'TS']


# Determine features values based on a window-sized sequence
def calculate_feature_values(s, hrpn, idx, gc):
        GH, structure = G_H(hrpn, True)
        hairpin_start, hairpin_end, loop_start, loop_end, hairpin_length, loop_length, total_stacking, total_unpaired, base_stacking, num_loops, consecutive_stacking, max_loop = get_structure_components(structure)
        hairpin = hrpn[hairpin_start:hairpin_end+1]
        U_tract = s[idx+hairpin_end+1:idx+hairpin_end+1+TAIL_LENGTH]
        A_tract = s[max(idx+hairpin_start-TAIL_LENGTH,0):idx+hairpin_start]
        features = {'hairpin_length':hairpin_length, 'loop_length':loop_length, 'total_stacking':total_stacking, 'total_unpaired':total_unpaired, 'base_stacking':base_stacking, 'num_loops':num_loops, 'consecutive_stacking':consecutive_stacking, 'max_loop':max_loop, 'GH':GH, 'structure':structure, 'hairpin_start':hairpin_start, 'hairpin_end':hairpin_end}
        features = calculate_feature_values_from_components(hairpin, U_tract, A_tract, gc, features, loop_start-hairpin_start, loop_end-hairpin_start)
        return features, hairpin, U_tract, A_tract


# Determine feature values based on a hairpin, U-tract, and A-tract sequences
def calculate_feature_values_from_components(hairpin, U_tract, A_tract, gc, features={}, loop_start=0, loop_end=0):
        if (len(features) == 0):
                GH, structure = G_H(hairpin, True)
                hairpin_start, hairpin_end, loop_start, loop_end, hairpin_length, loop_length, total_stacking, total_unpaired, base_stacking, num_loops, consecutive_stacking, max_loop = get_structure_components(structure)
                features = {'hairpin_length':hairpin_length, 'loop_length':loop_length, 'total_stacking':total_stacking, 'total_unpaired':total_unpaired, 'base_stacking':base_stacking, 'num_loops':num_loops, 'consecutive_stacking':consecutive_stacking, 'max_loop':max_loop, 'GH':GH, 'structure':structure, 'hairpin_start':hairping_start, 'hairpin_end':hairpin_end}

        GH_avg = features['GH'] / features['hairpin_length'] if (features['hairpin_length'] > 0) else 0.0
        GU = G_U(U_tract)
        if (hairpin == ''):
                GB = 0.0
                GA = 0.0
                TT_hairpin, hairpin_current, structure_TT = 0.0, hairpin, ''
        else:
                GB = G_B(hairpin)
                GA = G_A(A_tract, hairpin[0], hairpin[-1], U_tract)
                TT_hairpin, hairpin_current, structure_TT = hairpin_TT(hairpin, len(hairpin)-1, True)
        GL = G_L(hairpin[loop_start:loop_end+1])
        TS = T_S(GU, GL, GA, GB)
        features['GH_avg'], features['GU'], features['GB'] = GH_avg, GU, GB
        features['GA'], features['GL'], features['TS'] = GA, GL, TS
        
        hairpin_current = hairpin_current.replace(' ', '')
        structure_TT = structure_TT.replace(' ', '')
        TT_tail = tail_TT(U_tract, 0)
        TT_score = score(gc, TT_hairpin, TT_tail)
        hairpin_start_TT, hairpin_end_TT, loop_start_TT, loop_end_TT, hairpin_length_TT, loop_length_TT, total_stacking_TT, total_unpaired_TT, base_stacking_TT, num_loops_TT, consecutive_stacking_TT, max_loop_TT = get_structure_components(structure_TT)
        TT_hairpin_avg = TT_hairpin / hairpin_length_TT if (hairpin_length_TT > 0) else 0.0
        features['TT_hairpin'], features['TT_hairpin_avg'] = TT_hairpin, TT_hairpin_avg
        features['TT_tail'], features['TT_score'] = TT_tail, TT_score
        features['hairpin_length_TT'], features['loop_length_TT'] = hairpin_length_TT, loop_length_TT
        features['total_stacking_TT'], features['total_unpaired_TT'] = total_stacking_TT, total_unpaired_TT
        features['base_stacking_TT'], features['num_loops_TT'] = base_stacking_TT, num_loops_TT
        features['consecutive_stacking_TT'], features['max_loop_TT'] = consecutive_stacking_TT, max_loop_TT
        features['TT_structure'] = structure_TT
        return features


# RETURN DICTIONARY OF FEATURE VALUES FOR SINGLE BEST TERMINATOR IN SEQUENCE
# USE TT SCORE TO DETERMINE BEST TERMINATOR
def scan_TT(sequence, gc):
        best_TT_score, best_index = -1, -1
        for j in range(0, max(len(sequence)-WINDOW+1, 1), STEP):
                hrpn = sequence[j:min(j+HAIRPIN_LENGTH, len(sequence)-TAIL_LENGTH)]
                TT_hairpin = hairpin_TT(hrpn, len(hrpn)-1, False)
                TT_tail = tail_TT(sequence[j+len(hrpn):], 0)
                TT_score = score(gc, TT_hairpin, TT_tail)
                if (TT_score > best_TT_score): best_TT_score, best_index = TT_score, j
        hrpn = sequence[best_index:min(best_index+HAIRPIN_LENGTH, len(sequence)-TAIL_LENGTH)]
        features, hairpin, U_tract, A_tract = calculate_feature_values(sequence, hrpn, best_index, gc)
        return features, hairpin, U_tract, A_tract


def scan_oneprocess_1(j):
        hrpn = SEQUENCE[j:min(j+HAIRPIN_LENGTH, len(SEQUENCE)-TAIL_LENGTH)]
        TT_hairpin = hairpin_TT(hrpn, len(hrpn)-1, False)
        TT_tail = tail_TT(SEQUENCE[j+len(hrpn):], 0)
        TT_score = score(GC_PERCENT, TT_hairpin, TT_tail)
        if (TT_score >= 30): return (hrpn, j, TT_score)
        else: return '', -1, -1


def scan_oneprocess_2(candidate):
        hrpn, j, TT_score = candidate
        features, hairpin, U_tract, A_tract = calculate_feature_values(SEQUENCE, hrpn, j, GC_PERCENT)
        return (features, hairpin, U_tract, A_tract, TT_score, j)


# DIFFERENT PROCESSES FOR DIFFERENT INDICES IN ONE SEQUENCE
def scan_multiprocess(sequence, model, scaler, gc, THREADS):
        global SEQUENCE, GC_PERCENT
        SEQUENCE, GC_PERCENT = sequence, gc
        indices = list(range(0, max(len(SEQUENCE)-WINDOW+1, 1), STEP))
        with multiprocessing.Pool(THREADS) as pool:
                candidates = pool.map(scan_oneprocess_1, indices)
        candidates2 = []
        for hrpn, j, TT_score in candidates:
                if (j >= 0): candidates2.append((hrpn, j, TT_score))
        with multiprocessing.Pool(THREADS) as pool:
                candidates3 = pool.map(scan_oneprocess_2, candidates2)
        predictions = []
        for features, hairpin, U_tract, A_tract, TT_score, j in candidates3:
                x = []
                for f in FEATURES: x.append(features[f])
                x = scaler.transform(np.array(x).reshape(1, -1))
                pred = model.predict_proba(x)[0,1]
                if (pred >= 0.5):
                        predictions.append((j+features['hairpin_start'], j+features['hairpin_end'], A_tract, hairpin, U_tract, features['structure'], features['GH'], pred))
        predictions.sort(key=itemgetter(1))
        return predictions


# RETURN LIST OF COORDS OF ALL TERMINATORS FOUND IN THE SEQUENCE
def scan(sequence, model, scaler, gc):
        predictions = []  # Inclusive [start, stop]
        for j in range(0, max(len(sequence)-WINDOW+1, 1), STEP):
                hrpn = sequence[j:min(j+HAIRPIN_LENGTH, len(sequence)-TAIL_LENGTH)]
                TT_hairpin = hairpin_TT(hrpn, len(hrpn)-1, False)
                TT_tail = tail_TT(sequence[j+len(hrpn):], 0)
                TT_score = score(gc, TT_hairpin, TT_tail)
                if (TT_score >= 30):
                        features, hairpin, U_tract, A_tract = calculate_feature_values(sequence, hrpn, j, gc)
                        x = []
                        for f in FEATURES: x.append(features[f])
                        x = scaler.transform(np.array(x).reshape(1, -1))
                        pred = model.predict_proba(x)[0,1]
                        if (pred >= 0.5):
                                predictions.append((j+features['hairpin_start'], j+features['hairpin_end'], A_tract, hairpin, U_tract, features['structure'], features['GH'], pred))
        return predictions


def scan_many_oneprocess(idx):
        return (idx, scan(SEQUENCES[idx], MODEL, SCALER, GC_PERCENT))


# DIFFERENT PROCESSES FOR DIFFERENT SEQUENCES
def scan_many_multiprocess(sequences, model, scaler, gc, THREADS):
        global SEQUENCES, MODEL, SCALER, GC_PERCENT
        SEQUENCES, MODEL, SCALER, GC_PERCENT = sequences, model, scaler, gc
        indices = list(range(len(SEQUENCES)))
        with multiprocessing.Pool(THREADS) as pool:
                candidates = pool.map(scan_many_oneprocess, indices)
        candidates.sort(key=itemgetter(0))
        return candidates


####################
#####   MAIN   #####
####################

if __name__=='__main__':
    sys.stderr.write('\nThe Features library is not meant to be executed.\nRather it provides useful helper functions for other Python programs.\n\n')



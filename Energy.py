
import sys, math
import numpy as np
from Parameters import *


######################
#####   GLOBAL   #####
######################

s1, s2, W, V, H = '', '', None, None, None
MULTIPLE_HAIRPIN = True
MAX_LOOP = 9999
MAX_SEQ_LENGTH = 100
V = np.zeros((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH))
W = np.zeros((MAX_SEQ_LENGTH, MAX_SEQ_LENGTH))


#####################
#####   UTILS   #####
#####################

IS_PAIR = {'GC', 'GT', 'CG', 'AT', 'TA', 'TG'}
def is_pair(a, b):
    if (a + b in IS_PAIR): return True
    return False


def hairpin(s1, i, j):
    if (not is_pair(s1[i], s1[j])): return math.inf  # Ensure loop has closing basepair

    # Get contribution to energy of loop length
    length_energy = math.inf
    length = j - i - 1
    if (length <= 0): return math.inf
    if (length < len(hairpin_loop_lengths)): length_energy = hairpin_loop_lengths[length]
    else: length_energy = hairpin_loop_lengths[-1] + (1.079*math.log(float(length)/float(len(hairpin_loop_lengths)-1)))
    
    # Get terminal stacking pairs contribution
    terminal_energy = 0.0
    terminal_seq = s1[i] + s1[j] + s1[i+1] + s1[j-1]
    if (length > 3): terminal_energy = hairpin_terminals[terminal_seq]  # Only apply for loops >= 3nts

    # Get bonus energy contribution
    bonus_energy = 0.0
    if (s1[i:j+1] in hairpin_bonuses): bonus_energy = hairpin_bonuses[s1[i:j+1]]

    # Get miscellaneous energy contribution
    miscellaneous_energy = 0.0  # Ignore these contributions (miscloop.dat file)
    if (i-2 >= 0) and (s1[i-2] == 'G') and (s1[i-1] == 'G') and (s1[i] == 'G') and (s1[j] == 'T'):
        miscellaneous_energy += -2.20  # GGG hairpin
    ss_loop = s1[i+1:j]  # Single stranded loop sequence
    if (len(ss_loop.replace('C','')) == 0):  # Single strand of loop contains all C's
        if (length == 3): miscellaneous_energy += 1.40
        else: miscellaneous_energy += 0.30*length + 1.60

    return length_energy + terminal_energy + bonus_energy + miscellaneous_energy

def stacking(i, j, x, y):
    seq = s1[i] + s2[j] + s1[x] + s2[y]
    if (seq in stacking_energies): return stacking_energies[seq]
    return math.inf

def bulge(i, j, x, y):
    min_bulge_length = x - i - 1
    max_bulge_length = max(j - y - 1, y - j - 1)  # One sequence or two
    if (min_bulge_length > max_bulge_length):
        temp = min_bulge_length
        min_bulge_length = max_bulge_length
        max_bulge_length = temp
    if (min_bulge_length != 0) or (max_bulge_length <= 0): return math.inf

    length_energy = math.inf
    if (max_bulge_length < len(bulge_loop_lengths)): length_energy = bulge_loop_lengths[max_bulge_length]
    else: length_energy = bulge_loop_lengths[-1] + (1.079*math.log(float(max_bulge_length)/float(len(bulge_loop_lengths)-1)))

    stacking_energy = 0.0  # Include stacking energy only for bulge of size 1
    if (max_bulge_length == 1): stacking_energy = stacking_energies[s1[i] + s2[j] + s1[x] + s2[y]]

    miscellaneous_energy = 0.0  # Miscellaneous energy
    if ((max_bulge_length > 1) and (s1[i] == 'A') and (s2[j] == 'T')) or ((max_bulge_length > 1) and (s1[i] == 'T') and (s2[j] == 'A')): miscellaneous_energy = 0.5
    return length_energy + stacking_energy + miscellaneous_energy

def interior(i, j, x, y):
    min_interior_length = x - i - 1
    max_interior_length = max(j - y - 1, y - j - 1)  # One sequence or two
    if (min_interior_length > max_interior_length):
        temp = min_interior_length
        min_interior_length = max_interior_length
        max_interior_length = temp
    if (min_interior_length == 0): return math.inf

    # 1x1 interior loop (special case)
    if (min_interior_length == 1) and (max_interior_length == 1):
        if (s1 == s2): seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j-1]
        else: seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j+1]
        if (seq in interior_1_1): return interior_1_1[seq]
        else: return math.inf

    # 1x2 interior loop (special case)
    if (min_interior_length == 1) and (max_interior_length == 2):
        # Assume, initially, that we have a 1x2 loop as opposed to a 2x1 loop
        if (s1 == s2):
            seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j-1] + s2[j-2]
            if (x-i > j-y):  # We have a 2x1 loop
                seq = s2[y] + s1[x] + s2[j] + s1[i] + s2[j-1] + s1[x-1] + s1[x-2]
        else:
            seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j+1] + s2[j+2]
            if (x-i > y-j):
                seq = s2[y] + s1[x] + s2[j] + s1[i] + s2[j+1] + s1[x-1] + s1[x-2]
        if (seq in interior_1_2): return interior_1_2[seq]
        else: return math.inf

    # 2x2 interior loop (special case)
    if (min_interior_length == 2) and (max_interior_length == 2):
        if (s1 == s2): seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j-1] + s1[i+2] + s2[j-2]
        else: seq = s1[i] + s2[j] + s1[x] + s2[y] + s1[i+1] + s2[j+1] + s1[i+2] + s2[j+2]
        if (seq in interior_2_2): return interior_2_2[seq]
        else: return math.inf

    # Length energy
    loop_length = max_interior_length + min_interior_length
    length_energy = math.inf
    if (loop_length < len(interior_loop_lengths)):
        length_energy = interior_loop_lengths[loop_length]
    else: length_energy = interior_loop_lengths[-1] * (1.079*math.log(float(loop_length)/float(len(interior_loop_lengths)-1)))

    # Terminal mismatch energies
    terminal_energy = 0.0
    if (s1 == s2): seq = s1[i] + s2[j] + s1[i+1] + s2[j-1]
    else: seq = s1[i] + s2[j] + s1[i+1] + s2[j+1]
    if (min_interior_length == 1) and (max_interior_length > 2):  # GAIL rule (substitute AA mismatch)
        seq = s1[i] + s2[j] + 'A' + 'A'
    terminal_energy += interior_terminals[seq]
    if (s1 == s2): seq = s2[y] + s1[x] + s2[y+1] + s1[x-1]
    else: seq = s2[y] + s1[x] + s2[y-1] + s1[x-1]
    if (min_interior_length == 1) and (max_interior_length > 2):  # GAIL rule (substitute AA mismatch)
        seq = s2[y] + s1[x] + 'A' + 'A'
    terminal_energy += interior_terminals[seq]

    # Asymmetry energy with branches N1 and N2 is the minimum of 3.0 or N*f(M),
    # where N = |N1-N2|, M is the minimum of 4, N1, and N2,
    # and f(1) = 0.4, f(2) = 0.3, f(3) = 0.2, and f(4) = 0.1
    # Taken from Jaeger, Turner, and Zuker (PNAS 1989)
    asymmetry_energy = 0.0
    N = max_interior_length - min_interior_length
    M = min(4, min_interior_length)
    if (M == 1): asymmetry_energy = N*0.4
    if (M == 2): asymmetry_energy = N*0.3
    if (M == 3): asymmetry_energy = N*0.2
    if (M == 4): asymmetry_energy = N*0.1
    asymmetry_energy = min(asymmetry_energy, 3.0)

    return length_energy + terminal_energy + asymmetry_energy


def output_array(M):
    if (s1 == s2):  # Structure for one sequence
        for i in range(M.shape[0]-1, -1, -1):
            sys.stdout.write(s1[i])
            for j in range(M.shape[1]):
                if (j < i): val = 'X'
                elif (M[i, j] > MAX_VALUE): val = '*'
                else: val = '{:.1f}'.format(M[i, j])
                sys.stdout.write('\t' + val)
            sys.stdout.write('\n')
        sys.stdout.write('\t' + '\t'.join(s1) + '\n')
        sys.stdout.write('\n')
    else:  # Hybridization for two sequences
        sys.stdout.write('\t' + '\t'.join(s2) + '\n')
        for i in range(M.shape[0]):
            sys.stdout.write(s1[i])
            for j in range(M.shape[1]):
                sys.stdout.write('\t' + '{:.1f}'.format(M[i, j]))
            sys.stdout.write('\n')
        sys.stdout.write('\n')


#########################################
#####   STRUCTURE OF ONE SEQUENCE   #####
#########################################

def compute_structure_with_minimum_energy(seq, return_structure=True, multiple_hairpin=True, max_loop=9999):
    global s1, s2, W, V, MULTIPLE_HAIRPIN, MAX_LOOP
    s1 = seq.upper().replace('U', 'T')
    s2 = s1
    if (s1.count('A') + s1.count('C') + s1.count('G') + s1.count('T') != len(s1)):
        sys.stderr.write('\nInput sequence contains non-nucleotide characters!\n\n'); sys.exit(1)
    #V = np.zeros((len(s1), len(s1)))
    #W = np.zeros((len(s1), len(s1)))
    MULTIPLE_HAIRPIN = multiple_hairpin
    MAX_LOOP = max_loop
    
    for i in range(len(s1)-1, -1, -1):
        for j in range(i, len(s1)):

            # Fill in V table entry
            if (j-i <= 3): V[i, j] = math.inf  # If sequence has length <= 4 then infinity
            else:  # Compute optimal energy for each subsequence
                caseV_1 = hairpin(s1, i, j)  # Hairpin loop
                caseV_2 = stacking(i, j, i+1, j-1) + V[i+1, j-1]  # Stacking region

                # Interior loop
                caseV_3 = math.inf
                for x in range(i+1, min(j-2, i+1+MAX_LOOP)):
                    for y in range(max(x+1,j-MAX_LOOP), j):
                        temp = interior(i, j, x, y) + V[x, y]
                        if (temp < caseV_3): caseV_3 = temp

                # Bulge loop (right side)
                caseV_4 = math.inf
                for y in range(max(i+2, j-1-MAX_LOOP), j-1):
                    temp = bulge(i, j, i+1, y) + V[i+1, y]
                    if (temp < caseV_4): caseV_4 = temp

                # Bulge loop (left side)
                caseV_5 = math.inf
                for x in range(i+2, min(j-1, i+2+MAX_LOOP)):
                    temp = bulge(i, j, x, j-1) + V[x, j-1]
                    if (temp < caseV_5): caseV_5 = temp

                # Bifurcation
                caseV_6 = math.inf
                if (MULTIPLE_HAIRPIN):
                    for x in range(i+2, j-2):
                        temp = W[i+1, x] + W[x+1, j-1]
                        if (temp < caseV_6): caseV_6 = temp

                V[i, j] = min(caseV_1, caseV_2, caseV_3, caseV_4, caseV_5, caseV_6)

            # Fill in W table entry
            if (j-i <= 4): W[i, j] = math.inf  # If sequence has length <= 5 then infinity
            else:  # Compute optimal energy for each subsequence
                caseW_1 = W[i+1, j]  # i doesn't pair
                caseW_2 = W[i, j-1]  # j doesn't pair
                caseW_3 = V[i, j]    # i and j pair with each other

                # i and j pair, but not with each other
                caseW_4 = math.inf
                if (MULTIPLE_HAIRPIN):
                    for x in range(i+1, j-1):
                        temp = W[i, x] + W[x+1, j]
                        if (temp < caseW_4): caseW_4 = temp

                W[i, j] = min(caseW_1, caseW_2, caseW_3, caseW_4)

    if (W[0, len(s1)-1] >= 0):
        if (return_structure): return 0.0, '.' * len(s1)
    if (return_structure): return W[0, len(s1)-1], get_structure()
    else: return W[0, len(s1)-1], ''

def periods(n):
    return '.' * n

def get_structure():
    return get_structure_REC(W, 0, len(s1)-1, False)

# Backtrack through dynamic programming tables to create the *structure* with min energy.
# Recursively calculate the optimal structure for the subsequence between i and j,
# inclusive. Return the structure as a string.
MAX_VALUE = 999999999.9
def get_structure_REC(M, i, j, use_V):
    if (M[i][j] > MAX_VALUE): return periods(j-i+1)

    # V matrix
    # Determine where current table entry came from (i.e., backtrack)
    if (use_V):
        caseV_1 = hairpin(s1, i, j)  # Hairpin loop
        caseV_2 = stacking(i, j, i+1, j-1) + V[i+1, j-1]  # Stacking region

        # Interior loop
        caseV_3 = math.inf
        interiorX = -1
        interiorY = -1
        for x in range(i+1, min(j-2, i+1+MAX_LOOP)):
            for y in range(max(x+1, j-MAX_LOOP), j):
                temp = interior(i, j, x, y) + V[x, y]
                if (temp < caseV_3):
                    caseV_3 = temp
                    interiorX = x
                    interiorY = y

        # Bulge loop (right side)
        caseV_4 = math.inf
        bulgeY = -1
        for y in range(max(i+2, j-1-MAX_LOOP), j-1):
            temp = bulge(i, j, i+1, y) + V[i+1, y]
            if (temp < caseV_4):
                caseV_4 = temp
                bulgeY = y

        # Bulge loop (left side)
        caseV_5 = math.inf
        bulgeX = -1
        for x in range(i+2, min(j-1, i+2+MAX_LOOP)):
            temp = bulge(i, j, x, j-1) + V[x, j-1]
            if (temp < caseV_5):
                caseV_5 = temp
                bulgeX = x

        # Bifurcation
        caseV_6 = math.inf
        if (MULTIPLE_HAIRPIN):
            bifurcationX = -1
            for x in range(i+2, j-2):
                temp = W[i+1, x] + W[x+1, j-1]
                if (temp < caseV_6):
                    caseV_6 = temp
                    bifurcationX = x

        if (caseV_1 == V[i, j]):
            return '(' + periods(j-i-1) + ')'
        if (caseV_2 == V[i, j]):
            return '(' + get_structure_REC(V, i+1, j-1, True) + ')'
        if (caseV_3 == V[i, j]):
            return '(' + periods(interiorX-i-1) + get_structure_REC(V, interiorX, interiorY, True) + periods(j-interiorY-1) + ')'
        if (caseV_4 == V[i, j]):
            return '(' + get_structure_REC(V, i+1, bulgeY, True) + periods(j-bulgeY-1) + ')'
        if (caseV_5 == V[i, j]):
            return '(' + periods(bulgeX-i-1) + get_structure_REC(V, bulgeX, j-1, True) + ')'
        if (caseV_6 == V[i, j]):
            return '(' + get_structure_REC(W, i+1, bifurcationX, False) + get_structure_REC(W, bifurcationX+1, j-1, False) + ')'

    # W matrix
    # Determine where current table entry came from (i.e., backtrack)
    if (not use_V):
        caseW_1 = W[i+1, j]  # i doesn't pair
        caseW_2 = W[i, j-1]  # j doesn't pair
        caseW_3 = V[i, j]    # i and j pair with each other

        # i and j pair, but not with each other
        caseW_4 = math.inf
        if (MULTIPLE_HAIRPIN):
            pairX = -1
            for x in range(i+1, j-1):
                temp = W[i, x] + W[x+1, j]
                if (temp < caseW_4):
                    caseW_4 = temp
                    pairX = x

        if (caseW_1 == W[i, j]): return '.' + get_structure_REC(W, i+1, j, False)
        if (caseW_2 == W[i, j]): return get_structure_REC(W, i, j-1, False) + '.'
        if (caseW_3 == W[i, j]): return get_structure_REC(V, i, j, True)
        if (caseW_4 == W[i, j]): return get_structure_REC(W, i, pairX, False) + get_structure_REC(W, pairX+1, j, False)
    return ''


##############################################
#####   HYBRIDIZATION OF TWO SEQUENCES   #####
##############################################

def compute_hybridization_with_minimum_energy(seq1, seq2, return_hybridization=True):
    global s1, s2, H
    s1 = seq1.upper().replace('U', 'T')
    s2 = seq2.upper().replace('U', 'T')[::-1]  # Reverse second sequence!
    if (s1.count('A') + s1.count('C') + s1.count('G') + s1.count('T') != len(s1)):
        sys.stderr.write('\nInput sequence contains non-nucleotide characters!\n\n'); sys.exit(1)
    if (s2.count('A') + s2.count('C') + s2.count('G') + s2.count('T') != len(s2)):
        sys.stderr.write('\nInput sequence contains non-nucleotide characters!\n\n'); sys.exit(1)
    H = np.zeros((len(s1), len(s2)))
    optimal_energy, optimal_row, optimal_column = 0.0, -1, -1

    for i in range(len(s1)):
        for j in range(len(s2)):

            if (i == 0) or (j == 0): H[i, j] = 0.0  # First row or first column
            else:  # Compute optimal energy for each pair of subsequences
                caseH_1 = stacking(i-1, j-1, i, j) + H[i-1, j-1]  # Stacking region

                # Bulge loop (first sequence)
                caseH_2 = math.inf
                for x in range(i-1):
                    temp = bulge(x, j-1, i, j) + H[x, j-1]
                    if (temp < caseH_2): caseH_2 = temp

                # Bulge loop (second sequence)
                caseH_3 = math.inf
                for y in range(j-1):
                    temp = bulge(i-1, y, i, j) + H[i-1, y]
                    if (temp < caseH_3): caseH_3 = temp

                # Interior loop
                caseH_4 = math.inf
                for x in range(i-1):
                    for y in range(j-1):
                        temp = interior(x, y, i, j) + H[x, y]
                        if (temp < caseH_4): caseH_4 = temp

                caseH_5 = 0.0
                H[i, j] = min(caseH_1, caseH_2, caseH_3, caseH_4, caseH_5)
                if (H[i, j] <= optimal_energy):  # Optimal energy is min in table
                    optimal_energy = H[i, j]
                    optimal_row = i
                    optimal_column = j
    s1_hybrid, s2_hybrid, hybrid = '', '', ''
    if (return_hybridization): s1_hybrid, s2_hybrid, hybrid = get_hybridization(optimal_row, optimal_column)
    return optimal_energy, s1_hybrid, s2_hybrid, hybrid


def gaps(n):
    return '-' * n


def spaces(n):
    return ' ' * n


def get_hybridization(optimal_row, optimal_column):
    s1_hybrid = s1[optimal_row+1:]
    s2_hybrid = s2[optimal_column+1:]
    hybrid = spaces(max(len(s1)-optimal_row-1, len(s2)-optimal_column-1))
    return get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, optimal_row, optimal_column)


def get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, i, j):
    if (H[i, j] == 0.0):
        max_seq = max(i, j)
        s1_hybrid = spaces(max_seq-i) + s1[:i+1] + s1_hybrid
        s2_hybrid = spaces(max_seq-j) + s2[:j+1] + s2_hybrid
        if (i == optimal_row) and (j == optimal_column): hybrid = spaces(max(i+1, j+1)) + hybrid
        else: hybrid = spaces(max_seq) + '|' + hybrid
        return s1_hybrid, s2_hybrid, hybrid

    # Determine where current table entry came from
    caseH_1 = stacking(i-1, j-1, i, j) + H[i-1, j-1]  # Stacking region

    # Bulge loop (first sequence)
    caseH_2 = math.inf
    bulgeX = -1
    for x in range(i-1):
        temp = bulge(x, j-1, i, j) + H[x, j-1]
        if (temp < caseH_2):
            caseH_2 = temp
            bulgeX = x

    # Bulge loop (second sequence)
    caseH_3 = math.inf
    bulgeY = -1
    for y in range(j-1):
        temp = bulge(i-1, y, i, j) + H[i-1, y]
        if (temp < caseH_3):
            caseH_3 = temp
            bulgeY = y

    # Interior loop
    caseH_4 = math.inf
    interiorX = -1
    interiorY = -1
    for x in range(i-1):
        for y in range(j-1):
            temp = interior(x, y, i, j) + H[x, y]
            if (temp < caseH_4):
                caseH_4 = temp
                interiorX = x
                interiorY = y

    caseH_5 = 0.0
    if (caseH_1 == H[i, j]):
        s1_hybrid = s1[i] + s1_hybrid
        s2_hybrid = s2[j] + s2_hybrid
        hybrid = '|' + hybrid
        return get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, i-1, j-1)
    elif (caseH_2 == H[i, j]):
        s1_hybrid = s1[bulgeX+1:i] + s1[i] + s1_hybrid
        s2_hybrid = gaps(i-bulgeX-1) + s2[j] + s2_hybrid
        hybrid = spaces(i-bulgeX-1) + '|' + hybrid
        return get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, bulgeX, j-1)
    elif (caseH_3 == H[i, j]):
        s1_hybrid = gaps(j-bulgeY-1) + s1[i] + s1_hybrid
        s2_hybrid = s2[bulgeY+1:j] + s2[j] + s2_hybrid
        hybrid = spaces(j-bulgeY-1) + '|' + hybrid
        return get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, i-1, bulgeY)
    elif (caseH_4 == H[i, j]):
        max_interior = max(i-interiorX-1, j-interiorY-1)
        s1_hybrid = s1[interiorX+1:i] + gaps(max(max_interior-(i-interiorX-1), 0)) + s1[i] + s1_hybrid
        s2_hybrid = s2[interiorY+1:j] + gaps(max(max_interior-(j-interiorY-1), 0)) + s2[j] + s2_hybrid
        hybrid = spaces(max_interior) + '|' + hybrid
        return get_hybridization_REC(s1_hybrid, s2_hybrid, hybrid, optimal_row, optimal_column, interiorX, interiorY)
    elif (caseH_5 == H[i, j]):
        sys.stderr.write('Error - this case should never be reached.\n')
        max_seq = max(i, j)
        s1_hybrid = spaces(max_seq-i) + s1[:i+1] + s1_hybrid
        s2_hybrid = spaces(max_seq-j) + s2[:j+1] + s2_hybrid
        hybrid = spaces(max_seq) + '|' + hybrid
    else: sys.stderr.write('Error - this case should never be reached.\n')
    return s1_hybrid, s2_hybrid, hybrid


####################
#####   MAIN   #####
####################

if __name__ == '__main__':

    if (len(sys.argv) < 2):
        sys.stderr.write('\nUSAGE: python Energy.py <sequence> <OPT: sequence2>\n\n')
        sys.stderr.write('Energy takes either one RNA sequence or two RNA sequences as input. If one sequence, it computes the secondary structure with optimal energy. If two sequences, it computes the hybridization with optimal energy. Output is to stdout.\n\n')
        sys.exit(1)

    if (len(sys.argv) == 2):  # One sequence. Compute secondary structure.
        s1 = sys.argv[1]
        energy, structure = compute_structure_with_minimum_energy(s1, multiple_hairpin=True)
        sys.stdout.write('\nStructure with lowest energy:\n\n')
        sys.stdout.write('\t' + structure + '\n')
        sys.stdout.write('\t' + s1 + '\n\n')
        sys.stdout.write('Energy of most favorable structure:\t' + str(energy) + '\n\n')
        #output_array(V)
        #output_array(W)
    else:  # Two sequences. Compute hybridization.
        s1, s2 = sys.argv[1], sys.argv[2]
        energy, s1_hybrid, s2_hybrid, hybrid = compute_hybridization_with_minimum_energy(s1, s2)
        sys.stdout.write('\nHybridization with lowest energy:\n\n')
        sys.stdout.write('\t' + s1_hybrid + '\n')
        sys.stdout.write('\t' + hybrid + '\n')
        sys.stdout.write('\t' + s2_hybrid + '\n\n')
        sys.stdout.write('Optimal hybridization energy:\t' + str(energy) + '\n\n')


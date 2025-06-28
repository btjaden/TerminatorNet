
import sys, os, pickle, gzip
from operator import itemgetter
from Features import *


##########################
#####   PARAMETERS   #####
##########################

MODEL_NAME = 'model.pkl'
MODEL, SCALER = None, None
THREADS = 1
OUT = sys.stdout
SEQUENCES, SEQUENCE_NAMES, SEQUENCES_RC = [], [], []
GC = 0.5
CONTAINS_LONG_SEQUENCE = False
BOTH_STRANDS = False


#########################
#####   FUNCTIONS   #####
#########################

def usage():
        sys.stderr.write('\nTerminatorNet\n')
        sys.stderr.write('Version 1.0\n')
        sys.stderr.write('TerminatorNet identifies transcription terminators in prokaryotic genomic sequences\n\n')
        sys.stderr.write('EXAMPLE USAGE:   python TerminatorNet.py -f *.fa\n')
        sys.stderr.write('EXAMPLE USAGE:   python TerminatorNet.py -f *.fa -m model.pkl -n 8\n')
        sys.stderr.write('\n*****   Required argument   *****\n\n')
        sys.stderr.write('\t-f STRING\tFile of genomic sequences either in FASTA format\n')
        sys.stderr.write('\t\t\t\tor with each sequence separated by a blank line\n')
        sys.stderr.write('\n*****   Optional arguments   *****\n\n')
        sys.stderr.write('\t-m String\tFile containing the trained model in pickle format\n')
        sys.stderr.write('\t\t\t\t(default is model.pkl)\n')
        sys.stderr.write('\t-o String\tFile to which results should be output\n')
        sys.stderr.write('\t\t\t\t(default is standard out)\n')
        sys.stderr.write('\t-n Integer\tNumber of processes to use\n')
        sys.stderr.write('\t\t\t\t(default is 1)\n')
        sys.stderr.write('\t-s       \tSearch both strands of sequence(s)\n')
        sys.stderr.write('\t\t\t\t(default is to search only forward strand)\n')
        sys.stderr.write('\t-h\t\tprint USAGE and DESCRIPTION, ignore all other flags\n')
        sys.stderr.write('\t-help\t\tprint USAGE and DESCRIPTION, ignore all other flags\n')
        sys.stderr.write('\n')
        sys.exit(1)


def set_parameters(model_name=MODEL_NAME, threads=THREADS, out=OUT, gc=GC, sequences=SEQUENCES, sequence_names=SEQUENCE_NAMES, sequence_filename=None, both_strands=BOTH_STRANDS):
        global MODEL, SCALER, THREADS, OUT, GC, BOTH_STRANDS, SEQUENCES, SEQUENCE_NAMES, SEQUENCES_RC, CONTAINS_LONG_SEQUENCE
        if (not os.path.exists(model_name)): sys.stderr.write('\nError - could not find model file ' + model_name + '\n\n'); sys.exit(1)
        try:
                with open(model_name, 'rb') as f: MODEL, SCALER = pickle.load(f)
        except: sys.stderr.write('\nError - could not read in pickled model file ' + model_name + '\n\n'); sys.exit(1)
        OUT = open(out, 'w') if isinstance(out, str) else out
        THREADS = threads
        GC = gc
        BOTH_STRANDS = both_strands

        # Process either a list of sequences or a file containing sequences
        if (len(sequences) > 0):  # We are given a list of sequences
                SEQUENCES = sequences
                if (len(sequence_names) != len(sequences)):
                        sequence_names = []
                        for i in range(len(sequences)): sequence_names.append('Seq_' + str(i+1))
                SEQUENCE_NAMES = sequence_names
        else:
                if (sequence_filename is None): sys.stderr.write('\nError - either a list of sequences or a file containing sequences must be provided\n\n'); sys.exit(1)
                if (not os.path.exists(sequence_filename)): sys.stderr.write('\nError - could not find sequence file ' + sequence_filename + '\n\n'); sys.exit(1)
                SEQUENCES, SEQUENCE_NAMES = read_in_sequences(sequence_filename)
                if (len(SEQUENCES) == 0): sys.stderr.write('\nError - could not read in genomic sequences from file ' + sequence_filename + '\n\n'); sys.exit(1)
        SEQUENCES, GC, CONTAINS_LONG_SEQUENCE = clean_sequences(SEQUENCES)
        GC = 0.5  # Always use GC of 0.5
        if (BOTH_STRANDS): SEQUENCES_RC = reverse_complement_sequences(SEQUENCES)


def command_line_arguments():
        global MODEL, SCALER, THREADS, OUT, SEQUENCES, SEQUENCE_NAMES, GC, CONTAINS_LONG_SEQUENCE, BOTH_STRANDS, SEQUENCES_RC
        MODEL, SCALER, THREADS, OUT, SEQUENCES, SEQUENCE_NAMES, GC, CONTAINS_LONG_SEQUENCE, BOTH_STRANDS, SEQUENCES_RC = None, None, 1, sys.stdout, [], [], 0.5, False, False, []
        if (len(sys.argv) < 2): usage(); sys.exit(1)
        i = 0
        while (i < len(sys.argv)):
                if ('TerminatorNet.py' in sys.argv[i]): i+= 1; continue
                elif ('-h' in sys.argv[i].lower()) or ('-help' in sys.argv[i].lower()): usage()
                elif ('-m' in sys.argv[i].lower()):  # Model file
                        if (i == len(sys.argv)-1) or (not os.path.exists(sys.argv[i+1])): sys.stderr.write('\nError - expecting pickled model file to follow -m command line argument\n\n'); sys.exit(1)
                        try:
                                with open(sys.argv[i+1], 'rb') as f: MODEL, SCALER = pickle.load(f)
                        except: sys.stderr.write('\nError - expecting pickled model file to follow -m command line argument\n\n'); sys.exit(1)
                        i += 1
                elif ('-n' in sys.argv[i].lower()):  # Number of processes
                        if (i == len(sys.argv)-1): sys.stderr.write('\nError - expecting an integer to follow the -n command line argument indicating the number of processes to be used\n\n'); sys.exit(1)
                        try: THREADS = int(sys.argv[i+1])
                        except: sys.stderr.write('\nError - expecting an integer to follow the -n command line argument indicating the number of processes to be used\n\n'); sys.exit(1)
                        i += 1
                elif ('-f' in sys.argv[i].lower()):
                        if (i == len(sys.argv)-1) or (not os.path.exists(sys.argv[i+1])): sys.stderr.write('\nError - expecting file of genomic sequences to follow -f command line argument\n\n'); sys.exit(1)
                        SEQUENCES, SEQUENCE_NAMES = read_in_sequences(sys.argv[i+1])
                        if (len(SEQUENCES) == 0): sys.stderr.write('\nError - could not read in genomic sequences from file ' + sys.argv[i+1] + '\n\n'); sys.exit(1)
                        SEQUENCES, GC, CONTAINS_LONG_SEQUENCE = clean_sequences(SEQUENCES)
                        GC = 0.5  # Always use GC of 0.5
                        i += 1
                elif ('-o' in sys.argv[i].lower()):
                        if (i == len(sys.argv)-1): sys.stderr.write('\nError - expecting name of file to follow -o command line argument\n\n'); sys.exit(1)
                        OUT = open(sys.argv[i+1], 'w')
                        i += 1
                elif ('-s' in sys.argv[i].lower()): BOTH_STRANDS = True
                else: usage()
                i += 1
        if (len(SEQUENCES) == 0):
                sys.stderr.write('Error - the command line argument -f is required followed by the name of a file containing one or more genomic sequences\n\n'); sys.exit(1)
        if (MODEL == None) and (not os.path.exists('model.pkl')):
                sys.stderr.write('Error - unable to load model, please ensure the file model.pkl is in this directory or use the -m command line argument to specify the path to a model file in pickle format\n\n'); sys.exit(1)
        elif (MODEL == None) and (os.path.exists('model.pkl')):
                with open('model.pkl', 'rb') as f: MODEL, SCALER = pickle.load(f)
        if (BOTH_STRANDS): SEQUENCES_RC = reverse_complement_sequences(SEQUENCES)


def read_in_sequences(filename):
        sequences, sequence_names, in_file = [], [], None
        try:
                if (filename.endswith('.gz')) or (filename.endswith('.gzip')):
                        in_file = gzip.open(filename, 'rt')
                else: in_file = open(filename, 'r')

                line = in_file.readline()
                while (line == ''): line = in_file.readline()  # Ignore blank header lines
                if (line.startswith('>')):  # FASTA file
                        s, s_name = '', ''
                        while (line != ''):
                                if (line.startswith('>')):
                                        if (len(s) > 0):
                                                sequences.append(s)
                                                if (s_name == ''): s_name = 'Seq_' + str(len(sequences))
                                                sequence_names.append(s_name)
                                                s = ''
                                        s_name = line[1:].strip()
                                else: s += line.strip()
                                line = in_file.readline()
                        if (len(s) > 0):
                                sequences.append(s)
                                if (s_name == ''): s_name = 'Seq_' + str(len(sequences))
                                sequence_names.append(s_name)
                else:  # File where sequences are separated by blank lines
                        s = ''
                        while (line != ''):
                                if (line.strip() == ''):
                                        if (len(s) > 0):
                                                sequences.append(s)
                                                sequence_names.append('Seq_' + str(len(sequences)))
                                                s = ''
                                else: s += line.strip()
                                line = in_file.readline()
                        if (len(s) > 0): sequences.append(s); sequence_names.append('Seq_' + str(len(sequences)))
                in_file.close()
        except:
                if (not in_file is None): in_file.close()
        return sequences, sequence_names


def clean_sequences(sequences):
        LONG_SEQUENCE = 501  # Below this length, do not multi-processing single sequence
        VALID_NT_SEQUENCE_THRESHOLD = 0.75  # Sequences must contain at least this % NTs
        CONTAINS_LONG_SEQUENCE = False
        gc_count, total_count = 0, 0
        for i in range(len(sequences)):
                if (len(str(sequences[i])) < 4): sequences[i] = ''; continue
                s = sequences[i].replace(' ','').upper().replace('U','T')
                gc = s.count('G') + s.count('C')
                at = s.count('A') + s.count('T')
                if (gc + at < VALID_NT_SEQUENCE_THRESHOLD*len(s)):  # Too many non-NTs
                        sys.stderr.write('\n\nWARNING - provided sequence contains too many non-nucleotide characters\n\n')
                elif (gc + at != len(s)):  # Remove non-nucleotide characters
                        s2 = []
                        for ch in s:
                                if (ch not in 'ACGT'): s2.append('')
                                else: s2.append(ch)
                        s = ''.join(s2)
                sequences[i] = s
                if (len(s) >= LONG_SEQUENCE): CONTAINS_LONG_SEQUENCE = True
                gc_count += gc
                total_count += gc + at
        if (total_count < 100): return sequences, 0.5, CONTAINS_LONG_SEQUENCE  # Use GC of 50%
        return sequences, float(gc_count)/float(total_count), CONTAINS_LONG_SEQUENCE


def reverse_complement_sequences(sequences):
        sequences_RC = []
        for s in sequences: sequences_RC.append(reverse_complement(s))
        return sequences_RC


def merge_overlapping(terms, seq_length=0, strand='+'):
        terms_non_overlapping = []
        index = 0
        while (index < len(terms)):
                best_start, best_stop, best_A_tract, best_hairpin, best_U_tract, best_structure, best_GH, best_pred = terms[index]
                j = index + 1
                while (j < len(terms)) and (terms[j][0] <= best_stop):
                        start, stop, A_tract, hairpin, U_tract, structure, GH, pred = terms[j]
                        if (pred >= best_pred): best_start, best_stop, best_A_tract, best_hairpin, best_U_tract, best_structure, best_GH, best_pred = start, stop, A_tract, hairpin, U_tract, structure, GH, pred
                        j += 1
                best_structure = best_structure[best_structure.find('('):best_structure.rfind(')')+1]
                terms_non_overlapping.append([best_A_tract, best_hairpin, best_U_tract, best_structure, best_GH, best_pred, best_start+1, best_stop+1])  # Add 1 to coords so they are not 0-indexed
                index = j
                if (BOTH_STRANDS):  # Handle case we both strands are searched
                        term = terms_non_overlapping[-1]
                        if (strand == '-'):  # Update reverse complement coords
                                stop_RC = seq_length - term[-2]
                                start_RC = seq_length - term[-1]
                                term[-2] = start_RC + 1
                                term[-1] = stop_RC + 1
                        term.append(strand)
        if (BOTH_STRANDS and (strand == '-')): terms_non_overlapping.sort(key=itemgetter(7))
        return terms_non_overlapping


def output_terms(terms, name):
        if (len(terms) == 0): OUT.write(name + '\t' + 'NO TERMINATORS FOUND' + '\n')
        else:
                for i in range(len(terms)):
                        OUT.write(name + '\t' + 'TERM_' + str(i+1) + '\t' + '\t'.join(map(str, list(terms[i]))) + '\n')


def run_TerminatorNet():
        if (THREADS == 1):
                for i in range(len(SEQUENCES)):
                        terms = scan(SEQUENCES[i], MODEL, SCALER, GC)
                        terms = merge_overlapping(terms)
                        output_terms(terms, SEQUENCE_NAMES[i])
                if (BOTH_STRANDS):
                        for i in range(len(SEQUENCES_RC)):
                                terms = scan(SEQUENCES_RC[i], MODEL, SCALER, GC)
                                terms = merge_overlapping(terms, len(SEQUENCES_RC[i]), '-')
                                output_terms(terms, SEQUENCE_NAMES[i])
        elif (len(SEQUENCES) == 1) and (not CONTAINS_LONG_SEQUENCE):  # Single short sequence
                terms = scan(SEQUENCES[0], MODEL, SCALER, GC)
                terms = merge_overlapping(terms)
                output_terms(terms, SEQUENCE_NAMES[0])
                if (BOTH_STRANDS):
                        terms = scan(SEQUENCES_RC[0], MODEL, SCALER, GC)
                        terms = merge_overlapping(terms, len(SEQUENCES_RC[0]), '-')
                        output_terms(terms, SEQUENCE_NAMES[0])
        elif (len(SEQUENCES) == 1) and (CONTAINS_LONG_SEQUENCE):  # Single long sequence
                terms = scan_multiprocess(SEQUENCES[0], MODEL, SCALER, GC, THREADS)
                terms = merge_overlapping(terms)
                output_terms(terms, SEQUENCE_NAMES[0])
                if (BOTH_STRANDS):
                        terms = scan_multiprocess(SEQUENCES_RC[0], MODEL, SCALER, GC, THREADS)
                        terms = merge_overlapping(terms, len(SEQUENCES_RC[0]), '-')
                        output_terms(terms, SEQUENCE_NAMES[0])
        elif (not CONTAINS_LONG_SEQUENCE):  # Multiple short sequence
                terms_many = scan_many_multiprocess(SEQUENCES, MODEL, SCALER, GC, THREADS)
                for i in range(len(terms_many)):
                        idx, terms = terms_many[i]
                        terms = merge_overlapping(terms)
                        output_terms(terms, SEQUENCE_NAMES[i])
                if (BOTH_STRANDS):
                        terms_many = scan_many_multiprocess(SEQUENCES_RC, MODEL, SCALER, GC, THREADS)
                        for i in range(len(terms_many)):
                                idx, terms = terms_many[i]
                                terms = merge_overlapping(terms, len(SEQUENCES_RC[i]), '-')
                                output_terms(terms, SEQUENCE_NAMES[i])
        elif (CONTAINS_LONG_SEQUENCE):  # Multiple long sequences
                for i in range(len(SEQUENCES)):
                        terms = scan_multiprocess(SEQUENCES[i], MODEL, SCALER, GC, THREADS)
                        terms = merge_overlapping(terms)
                        output_terms(terms, SEQUENCE_NAMES[i])
                if (BOTH_STRANDS):
                        for i in range(len(SEQUENCES_RC)):
                                terms = scan_multiprocess(SEQUENCES_RC[i], MODEL, SCALER, GC, THREADS)
                                terms = merge_overlapping(terms, len(SEQUENCES_RC[i]), '-')
                                output_terms(terms, SEQUENCE_NAMES[i])
        OUT.close()


####################
#####   MAIN   #####
####################

if __name__ == '__main__':

        command_line_arguments()
        run_TerminatorNet()



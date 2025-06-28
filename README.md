# <img src="https://cs.wellesley.edu/~btjaden/TermNet/img/Hairpin_A.png" width=100> [TerminatorNet](https://cs.wellesley.edu/~btjaden/TermNet) <img src="https://cs.wellesley.edu/~btjaden/TermNet/img/Hairpin_B.png" width=100>
==========

### [TerminatorNet](https://cs.wellesley.edu/~btjaden/TermNet) is a tool for predicting intrinsic ***transcription terminators*** in bacteria<BR><BR>

EXAMPLE USAGE: &nbsp;&nbsp;&nbsp;&nbsp;`python TerminatorNet.py -f *.fa`<BR>
EXAMPLE USAGE: &nbsp;&nbsp;&nbsp;&nbsp;`python TerminatorNet.py -f *.fa -m model.pkl -n 8`<BR>

*****   Required argument   *****

        -f    STRING    File of genomic sequences either in FASTA format
                               or with each sequence separated by a blank line


*****   Optional arguments  *****

        -m    STRING    File containing the trained model in pickle format
                              (default is model.pkl)
        -o    STRING    File to which results should be output
                              (default is standard out)
        -n    INTEGER   Number of processes to use
                              (default is 1)
        -s              Search both strands of sequence(s)
                              (default is to search only forward strand)
        -h              print USAGE and DESCRIPTION, ignore all other flags
        -help           print USAGE and DESCRIPTION, ignore all other flags

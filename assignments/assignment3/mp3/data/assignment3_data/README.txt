Distinguish letters in the ee-set (B,C,D,E,G,P,T,V,Z) from letters in the eh-set (F,L,M,N,S,X) using a logistic classifier.

Labels: are in the file "indexing.txt".  Each line of this file is of the form "label filename", where label is either "-1" (ee-set) or "1" (eh-set).  Note that, you can preprocess the label to 0/1 as necessary.

Features: ASCII text files in the directory "samples". each contain one 16-dimensional mel-frequency-cepstral vector, extracted from the peak of the vowel.
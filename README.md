EM-Clustering
=============

Implement EM method to cluster in Python.
EM method is a soft-thresholding method which allows each data point to belong to a cluster with a probability.
EM method iterates by maximizing likelihood function.

To run the program:

    python em-cluster.py input_file number_clusters label_availability

input_file should be in csv format.
number_clusters is defined by the user.
label_availability tells whether the true label of data point is available or not.

    Y or y - has label.
    
If the input file has label, the label should be the last column.
Under such condition, the program will calculate a purity score, defined as the maximum proportion of each true cluster.

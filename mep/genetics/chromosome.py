class Chromosome(object):
    """
    Level above Gene. Each chromosome is a fixed number of genes and constants. We can think of a chromosome as a
    program where each gene is a line of code in the program. Genes can reference the result of other genes by their
    line number (address) in the program/chromosome. The overall fitness (i.e. error) is equal to the best of the genes.
    """

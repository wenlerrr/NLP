from collections import Counter
A = Counter({'a':1, 'b':2, 'c':3})
A += Counter({'b':3, 'c':4, 'd':5})
print(A)

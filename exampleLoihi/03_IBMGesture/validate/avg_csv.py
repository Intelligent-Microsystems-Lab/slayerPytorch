
import sys
import csv
from itertools import chain
# with open(str(sys.argv[1]),'r') as handle:
#     reader = csv.reader(handle)
#     next(reader, None)
#     counter = 0
#     for row in reader:
#         average = sum([int(densities) for densities in row]) / len(row)
#         print (
#             "{counter} has average of {average}".format(counter=counter, average=average)
#         )
#         counter +=1

from functools import reduce
with open(str(sys.argv[1]),'r') as handle:
    reader = csv.reader(handle)
    lst = []
    for row in reader:
        lst.append([int(densities) for densities in row])
    next(reader, None)
    flatten = list(chain.from_iterable(lst))
    import pdb;pdb.set_trace()
 
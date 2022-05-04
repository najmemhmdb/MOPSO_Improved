import numpy as np

#Programmed By Yang Shang shang  Email：yangshang0308@gmail.com  GitHub: https://github.com/DevilYangS/codes
def sortrows(Matrix, order = "ascend"):
    # By default, the rows are sorted according to the value of the first column.
    # If the value of the first column is the same, the value of the second column is followed,
    # and so on, the sort result and the corresponding index are returned (Reason: list.sort()
    # only returns the sorted As a result, np.argsort() requires multiple sorting, among which.
    #  np.lexsort() The operation object of is equivalent to sortcols, first sort the column by the last row,
    #  then by the penultimate column, and so on. np.lexsort((d,c,b,a) to [a,b,c, d] for sorting,
    #  where a is a column of vectors)
    Matrix_temp = Matrix[:, ::-1] #Because np.lexsort() starts sorting from the last row by default,
    # the matrix needs to be reversed and transposed
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank,:] # Matrix[rank] 也可以
    return Sorted_Matrix, rank
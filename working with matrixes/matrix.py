import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        # TODO - your code here
        if self.h == 1:
            determinant == self.g[0][0]
        elif self.h == 2:
            determinant = (self.g[0][0] * self.g[1][1]) - (self.g[0][1] * self.g[1][0])

        return Matrix(determinant)
    
    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        trace = 0
        for i in range(self.h):
            trace += self.g[i][i]

        return trace

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        inverse = []
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        # TODO - your code here
        if self.h == 1:
            inverse.append([1 / self.g[0][0]])
        elif self.h == 2:
            #If the matrix is 2x2, check that the matrix is invertible
            if self.g[0][0] * self.g[1][1] == self.g[0][1] * self.g[1][0]:
                raise ValueError('The matrix is not invertible.')
            else:
                a = self.g[0][0]
                b = self.g[0][1]
                c = self.g[1][0]
                d = self.g[1][1]

                factor = 1 / (a * d - b * c)
                inverse = [[d, -b],[-c, a]]

                for i in range(len(inverse)):
                    for j in range(len(inverse[0])):
                        inverse[i][j] = factor * inverse[i][j]

        return Matrix(inverse)

    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        # TODO - your code here
        matrix_transpose = []
        for c in range(self.w):
            new_row = []
            for r in range(self.h):
                new_row.append(self.g[r][c])
            matrix_transpose.append(new_row)

        return Matrix(matrix_transpose)

    def is_square(self):
        return self.h == self.w

    def dot_product(self, other):
        result = 0
        for i in range(len(self)):
            result += self[i] * other[i]

        return result
    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        #   
        # TODO - your code here
        #
        matrix_Sum = []
        row = []
        for i in range(self.h):
            row = []
            for j in range(self.w):
                m_add= self.g[i][j] + other.g[i][j]
                row.append(m_add)
            matrix_Sum.append(row)

        return Matrix(matrix_Sum)

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #   
        # TODO - your code here
        #
        neg_matrix = []
        for i in range(self.h):
            row = []
            tmp = 0
            for j in range(self.w):
                row.append(tmp - self.g[i][j])
            neg_matrix.append(row)

        return Matrix(neg_matrix)

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #   
        # TODO - your code here
        #
        matrix_Sub = []
        row = []
        for i in range(self.h):
            row = []
            for j in range(self.w):
                m_sub= self.g[i][j] - other.g[i][j]
                row.append(m_sub)
            matrix_Sub.append(row)
        return Matrix(matrix_Sub)
    
    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #   
        # TODO - your code here
        #
        mul_Matrix = []
        transposeB = Matrix.T(other)

        for r1 in range(self.h):
            new_row = []
            for r2 in range(transposeB.h):
                dp = Matrix.dot_product(self.g[r1], transposeB[r2])
                new_row.append(dp)
            mul_Matrix.append(new_row)

        return Matrix(mul_Matrix)

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            #   
            # TODO - your code here
            #
            result_matrix=zeroes(self.h,self.w)
            for i in range(self.h):
                for j in range(self.w):
                    result_matrix[i][j]=self.g[i][j]*other
            return result_matrix
            
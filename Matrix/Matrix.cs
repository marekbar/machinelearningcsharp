using System;

namespace Matrices
{
    /// <summary>
    /// Represents matrix
    /// </summary>
    public class Matrix
    {
        /// <summary>
        /// Number of rows
        /// </summary>
        private readonly int rows;

        /// <summary>
        /// Number of columns
        /// </summary>
        private readonly int cols;

        /// <summary>
        /// Number of rows
        /// </summary>
        public int RowsCount => rows;

        /// <summary>
        /// Number of columns
        /// </summary>
        public int ColumnCount => cols;

        /// <summary>
        /// Internal matrix representation
        /// </summary>
        private readonly double[][] matrix;

        /// <summary>
        /// Determinates if matrix is square
        /// </summary>
        public bool IsSquare => rows == cols;

        /// <summary>
        /// Determinant of permutation matrix - for private use only
        /// </summary>
        private double detOfP = 1;

        /// <summary>
        /// Lower matrix
        /// </summary>
        public Matrix L;

        /// <summary>
        /// Upper matrix
        /// </summary>
        public Matrix U;

        /// <summary>
        /// Permutations vector
        /// </summary>
        private int[] PermutationVector;

        public Matrix Transposed
        {
            get
            {
                try
                {
                    int row, col;
                    var mat = new Matrix(cols, rows);
                    for (row = 0; row < rows; row++)
                    {
                        for (col = 0; col < cols; col++)
                        {
                            mat[col, row] = this[row, col];
                        }
                    }
                    return mat;
                }
                catch (Exception ex)
                {
                    throw new Exception("Transposition error.", ex);
                }
            }
        }

        public Matrix Inverted
        {
            get
            {
                if (L == null) MakeLU();

                var inv = new Matrix(rows, cols);

                for (int i = 0; i < rows; i++)
                {
                    var Ei = Matrix.Zeros(rows, 1);
                    Ei[i, 0] = 1;
                    var col = SolveWith(Ei);

                    for (int j = 0; j < rows; j++) inv[j, i] = col[j, 0];
                }

                return inv;
            }
        }

        /// <summary>
        /// Gets permutation matrix
        /// </summary>
        public Matrix PermutationMatrix
        {
            get
            {
                if (L == null) MakeLU();

                Matrix matrix = Zeros(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    matrix[PermutationVector[i], i] = 1;
                }
                return matrix;
            }
        }

        /// <summary>
        /// Gets matrix determinant
        /// </summary>
        public double Determinant
        {
            get
            {
                if (L == null) MakeLU();
                double det = detOfP;
                for (int i = 0; i < rows; i++) det *= U[i, i];
                return det;
            }
        }

        /// <summary>
        /// Gets matrix determinant
        /// </summary>
        public double Det => Determinant;

        /// <summary>
        /// Check if matrix is vertical vector - [n,0]
        /// </summary>
        public bool IsVerticalVector => cols == 1;

        /// <summary>
        /// Check if matrix is horizontal vector - [0,n]
        /// </summary>
        public bool IsHorizontalVector => rows == 1;

        /// <summary>
        /// Check if matrix is vector
        /// </summary>
        public bool IsVector => IsVerticalVector || IsHorizontalVector;

        /// <summary>
        /// Check if matrix is filled with zeros only
        /// </summary>
        public bool HasOnlyZeros
        {
            get
            {
                int row, col, counter = 0;
                for (row = 0; row < rows; row++)
                {
                    for (col = 0; col < cols; col++)
                    {
                        if (this[row, col] == 0)
                        {
                            counter++;
                        }
                    }
                }
                return counter == rows * cols;
            }
        }

        public Matrix(int rows, int cols)
        {
            this.rows = rows;
            this.cols = cols;

            matrix = new double[rows][];

            for (var i = 0; i < rows; i++)
            {
                matrix[i] = new double[cols];
            }
        }

        public double this[int row, int column]
        {
            get => matrix[row][column];
            set => matrix[row][column] = value;
        }

        /// <summary>
        /// Get row
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public double[] this[int row] => matrix[row];

        /// <summary>
        /// Gets specified column
        /// </summary>
        /// <param name="column">int - column index</param>
        /// <returns>MatrixMB</returns>
        public Matrix GetColumn(int column)
        {
            var matrix = new Matrix(rows, 1);
            for (var i = 0; i < rows; i++)
            {
                matrix[i, 0] = this[i, column];
            }
            return matrix;
        }

        public void SetColumn(Matrix v, int column)
        {
            for (var i = 0; i < rows; i++)
            {
                this[i, column] = v[i, 0];
            }
        }

        public double MinValue
        {
            get
            {
                double min = this.matrix[0][0];
                int row, col;
                for (col = 0; col < cols; col++)
                {
                    for (row = 0; row < rows; row++)
                    {
                        if (this.matrix[row][col] < min) min = this.matrix[row][col];
                    }
                }
                return min;
            }
        }

        public double Sum
        {
            get
            {
                double sum = 0;
                int row, col;
                for (row = 0; row < rows; row++)
                {
                    for (col = 0; col < cols; col++)
                    {
                        sum += this[row, col];
                    }
                }

                return sum;
            }
        }

        public double MaxValue
        {
            get
            {
                double max = this.matrix[0][0];
                int row, col;
                for (col = 0; col < cols; col++)
                {
                    for (row = 0; row < rows; row++)
                    {
                        if (this.matrix[row][col] > max) max = this.matrix[row][col];
                    }
                }
                return max;
            }
        }

        public Matrix SetValuesTo(double value)
        {
            int row, col;
            for (col = 0; col < cols; col++)
            {
                for (row = 0; row < rows; row++)
                {
                    this.matrix[row][col] = value;
                }
            }
            return this;
        }

        public static Matrix Ones(int rows, int cols)
        {
            return new Matrix(rows, cols).SetValuesTo(1);
        }

        public static Matrix Zeros(int rows, int cols)
        {
            return new Matrix(rows, cols).SetValuesTo(0);
        }

        /// <summary>
        /// Gets Jacobian row 1 x n which is filled with zeros
        /// </summary>
        /// <param name="length">number of data in such row</param>
        /// <returns>Matrix horizontal vector</returns>
        public static Matrix JacobianRow(int length)
        {
            return Matrix.Zeros(1, length);
        }

        /// <summary>
        /// Creates matrix with diagonal values set to one
        /// </summary>
        /// <param name="rows">number of rows</param>
        /// <param name="cols">number of columns</param>
        /// <returns></returns>
        public static Matrix Identity(int rows, int cols)
        {
            Matrix m = new Matrix(rows, cols);
            for (int i = 0; i < System.Math.Min(rows, cols); i++)
            {
                m.matrix[i][i] = 1;
            }
            return m;
        }

        public static Matrix Random(int rows, int cols, int dispersion)
        {
            System.Random random = new System.Random();
            Matrix matrix = new Matrix(rows, cols);
            int max = dispersion;
            int min = -dispersion;
            int i, j;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    matrix[i, j] = random.NextDouble() * (max - min) + min;
                }
            }
            return matrix;
        }

        public Matrix Copy()
        {
            var m = new Matrix(this.RowsCount, this.ColumnCount);
            int i, j;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    m[i, j] = this[i, j];
                }
            }

            return m;
        }

        private Matrix MakeLU()
        {
            if (!IsSquare) throw new Exception("The matrix is not square!");
            L = Matrix.Identity(rows, cols);
            U = this.Copy();

            PermutationVector = new int[rows];
            for (int i = 0; i < rows; i++) PermutationVector[i] = i;

            double p = 0;
            double pom2;
            int k0 = 0;
            int pom1 = 0;

            for (int k = 0; k < rows - 1; k++)
            {
                p = 0;
                for (int i = k; i < rows; i++)      // find the row with the biggest pivot
                {
                    if (System.Math.Abs(U[i, k]) > p)
                    {
                        p = System.Math.Abs(U[i, k]);
                        k0 = i;
                    }
                }
                if (p == 0)
                {
                    throw new Exception("Making LU: matrix is singular!");
                }
                pom1 = PermutationVector[k];
                PermutationVector[k] = PermutationVector[k0];
                PermutationVector[k0] = pom1;

                for (int i = 0; i < k; i++)
                {
                    pom2 = L[k, i];
                    L[k, i] = L[k0, i];
                    L[k0, i] = pom2;
                }

                if (k != k0) detOfP *= -1;

                for (int i = 0; i < rows; i++)
                {
                    pom2 = U[k, i];
                    U[k, i] = U[k0, i];
                    U[k0, i] = pom2;
                }

                for (int i = k + 1; i < rows; i++)
                {
                    L[i, k] = U[i, k] / U[k, k];
                    for (int j = k; j < rows; j++)
                    {
                        U[i, j] -= L[i, k] * U[k, j];
                    }
                }
            }
        }

        /// <summary>
        /// Function solves Ax = b for A as a lower triangular matrix
        /// </summary>
        /// <param name="A">Matrix - lower triangular matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Matrix</returns>
        public static Matrix SubsForth(Matrix A, Matrix b)
        {
            if (A.L == null) A.MakeLU();
            var n = A.RowsCount;
            var x = new Matrix(n, 1);

            for (var i = 0; i < n; i++)
            {
                x[i, 0] = b[i, 0];
                for (var j = 0; j < i; j++) x[i, 0] -= A[i, j] * x[j, 0];
                x[i, 0] = x[i, 0] / A[i, i];
            }
            return x;
        }

        /// <summary>
        /// Function solves Ax = b for A as an upper triangular matrix
        /// </summary>
        /// <param name="A">Matrix - upper triangular matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Matrix</returns>
        public static Matrix SubsBack(Matrix A, Matrix b)
        {
            if (A.L == null) A.MakeLU();
            var n = A.RowsCount;
            var x = new Matrix(n, 1);

            for (var i = n - 1; i > -1; i--)
            {
                x[i, 0] = b[i, 0];
                for (var j = n - 1; j > i; j--) x[i, 0] -= A[i, j] * x[j, 0];
                x[i, 0] = x[i, 0] / A[i, i];
            }
            return x;
        }

        /// <summary>
        /// Function resolves Ax = v in confirmity with solution vector 
        /// </summary>
        /// <param name="vector">Matrix - solution vector</param>
        /// <returns>Matrix</returns>
        public Matrix SolveWith(Matrix vector)
        {
            if (rows != cols) throw new Exception("Solve: Matrix is not square.");
            if (rows != vector.RowsCount) throw new Exception("Solve: wrong number of results in solution vector.");
            if (L == null) MakeLU();

            var b = new Matrix(rows, 1);
            for (int i = 0; i < rows; i++) b[i, 0] = vector[PermutationVector[i], 0];   // switch two items in "v" due to permutation matrix

            var z = SubsForth(L, b);
            var x = SubsBack(U, z);

            return x;

        }

        public static Matrix operator +(Matrix matrix, double value)
        {
            int i, j;
            for (i = 0; i < matrix.RowsCount; i++)
            {
                for (j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] += value;
                }
            }
            return matrix;
        }

        public static Matrix operator /(Matrix matrix, double value)
        {
            int i, j;
            for (i = 0; i < matrix.RowsCount; i++)
            {
                for (j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] /= value;
                }
            }
            return matrix;
        }

        public static Matrix operator *(Matrix matrix, double value)
        {
            int i, j;
            for (i = 0; i < matrix.RowsCount; i++)
            {
                for (j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] *= value;
                }
            }
            return matrix;
        }

        public static Matrix operator -(Matrix matrix, double value)
        {
            int i, j;
            for (i = 0; i < matrix.RowsCount; i++)
            {
                for (j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] -= value;
                }
            }
            return matrix;
        }

        public static Matrix operator *(Matrix one, Matrix two)
        {
            var mat = new Matrix(one.RowsCount, two.ColumnCount);
            if (one.ColumnCount != two.RowsCount)
            {
                throw new Exception(
                    "Cannot multiply two matrices. Matrix one cols number doesn't match matrix two rows size. " +
                    $"Matrix one: {one.ColumnCount}x{one.RowsCount}, Matrix two: {two.ColumnCount}x{two.RowsCount}");
            }

            for (var i = 0; i < one.RowsCount; i++)
            {
                for (var j = 0; j < two.ColumnCount; j++)
                {
                    for (var k = 0; k < one.ColumnCount; k++)
                    {
                        mat[i, j] += one[i, k] * two[k, j];
                    }
                }
            }
            return mat;
        }

        public static Matrix operator /(Matrix one, Matrix two)
        {
            return one * two.Inverted;
        }

        public static Matrix operator +(Matrix one, Matrix two)
        {
            try
            {
                if (one.ColumnCount == two.ColumnCount && one.RowsCount == two.RowsCount)
                {
                    var matrix = new Matrix(one.RowsCount, one.ColumnCount);
                    int row, col;
                    for (row = 0; row < one.RowsCount; row++)
                    {
                        for (col = 0; col < one.ColumnCount; col++)
                        {
                            matrix[row, col] = one[row, col] + two[row, col];
                        }
                    }
                    return matrix;
                }

                throw new System.Exception(
                    $"Cannot add two matrices. Matrix size doesn't match. Matrix one: {one.ColumnCount}x{one.RowsCount}, Matrix two: {two.ColumnCount}x{two.RowsCount}");
            }
            catch (Exception ex)
            {
                throw new Exception("Matrix addition error", ex);
            }
        }

        public static Matrix operator -(Matrix one, Matrix two)
        {
            try
            {
                if (one.ColumnCount == two.ColumnCount && one.RowsCount == two.RowsCount)
                {
                    var matrix = new Matrix(one.RowsCount, one.ColumnCount);
                    int row, col;
                    for (row = 0; row < one.RowsCount; row++)
                    {
                        for (col = 0; col < one.ColumnCount; col++)
                        {
                            matrix[row, col] = one[row, col] - two[row, col];
                        }
                    }
                    return matrix;
                }
                else
                {
                    throw new System.Exception(System.String.Format("Cannot substract two matrices. Matrix size doesn't match. Matrix one: {0}x{1}, Matrix two: {2}x{3}", one.ColumnCount, one.RowsCount, two.ColumnCount, two.RowsCount));
                }
            }
            catch (System.Exception ex)
            {
                throw new Exception("Matrix addition error", ex);
            }
        }
    }
}

using System;
using System.Runtime.Remoting;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Matrices;
namespace ML.Matrices.Tests
{
    [TestClass]
    public class MatrixTests
    {
        [TestMethod]
        public void Create()
        {
            var m = new Matrix(2, 2);
            Assert.AreEqual(2, m.RowsCount);
            Assert.AreEqual(2, m.ColumnCount);
        }

        [TestMethod]
        public void Assign()
        {
            var m = new Matrix(2, 2)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [1, 0] = 3,
                [1, 1] = 4
            };

            Assert.AreEqual(1, m[0, 0]);
            Assert.AreEqual(2, m[0, 1]);
            Assert.AreEqual(3, m[1, 0]);
            Assert.AreEqual(4, m[1, 1]);
        }

        [TestMethod]
        public void MatrixMinValue()
        {
            var m = new Matrix(2, 2)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [1, 0] = 3,
                [1, 1] = 4
            };

            Assert.AreEqual(1, m.MinValue);
        }

        [TestMethod]
        public void MatrixMaxValue()
        {
            var m = new Matrix(2, 2)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [1, 0] = 3,
                [1, 1] = 4
            };

            Assert.AreEqual(4, m.MaxValue);
        }

        [TestMethod]
        public void MatrixIsSquare()
        {
            var m = new Matrix(2, 2)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [1, 0] = 3,
                [1, 1] = 4
            };

            Assert.IsTrue(m.IsSquare);
        }

        [TestMethod]
        public void MatrixSetValuesTo()
        {
            var m = new Matrix(2, 2)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [1, 0] = 3,
                [1, 1] = 4
            };
            m.SetValuesTo(0);
            Assert.AreEqual(0, m[0, 0]);
            Assert.AreEqual(0, m[0, 1]);
            Assert.AreEqual(0, m[1, 0]);
            Assert.AreEqual(0, m[1, 1]);
        }

        [TestMethod]
        public void MatrixZeros()
        {
            var m = Matrix.Zeros(2, 2);
            Assert.AreEqual(0, m[0, 0]);
            Assert.AreEqual(0, m[0, 1]);
            Assert.AreEqual(0, m[1, 0]);
            Assert.AreEqual(0, m[1, 1]);
        }

        [TestMethod]
        public void MatrixOnes()
        {
            var m = Matrix.Ones(2, 2);
            Assert.AreEqual(1, m[0, 0]);
            Assert.AreEqual(1, m[0, 1]);
            Assert.AreEqual(1, m[1, 0]);
            Assert.AreEqual(1, m[1, 1]);
        }

        [TestMethod]
        public void MatrixAsJacobianRow()
        {
            var j = Matrix.JacobianRow(3);
            Assert.AreEqual(1, j.RowsCount);
            Assert.AreEqual(3, j.ColumnCount);
            Assert.AreEqual(0, j[0, 0]);
            Assert.AreEqual(0, j[0, 1]);
            Assert.AreEqual(0, j[0, 2]);
        }

        [TestMethod]
        public void MatrixAsIdentity()
        {
            var m = Matrix.Identity(3, 3);
            Assert.AreEqual(1, m[0, 0]);
            Assert.AreEqual(0, m[0, 1]);
            Assert.AreEqual(0, m[0, 2]);
            Assert.AreEqual(0, m[1, 0]);
            Assert.AreEqual(1, m[1, 1]);
            Assert.AreEqual(0, m[1, 2]);
            Assert.AreEqual(0, m[2, 0]);
            Assert.AreEqual(0, m[2, 1]);
            Assert.AreEqual(1, m[2, 2]);
        }

        [TestMethod]
        public void MatrixGeneratedRandomly()
        {
            var m = Matrix.Random(2, 2, 10);
            Assert.AreEqual(2, m.RowsCount);
            Assert.AreEqual(2, m.ColumnCount);
            var p = m[0, 0] != m[0, 1];
            Assert.IsTrue(p);
            p = m[0, 1] != m[1, 0];
            Assert.IsTrue(p);
            p = m[1, 0] != m[1, 1];
            Assert.IsTrue(p);
        }

        [TestMethod]
        public void MatrixTransposition()
        {
            var notTransposed = new Matrix(2, 3)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [0, 2] = 3,
                [1, 0] = 4,
                [1, 1] = 5,
                [1, 2] = 6
            };

            var transposed = notTransposed.Transposed;
            Assert.AreEqual(3, transposed.RowsCount);
            Assert.AreEqual(2, transposed.ColumnCount);
            Assert.AreEqual(transposed[0, 0], 1);
            Assert.AreEqual(transposed[0, 1], 4);
            Assert.AreEqual(transposed[1, 0], 2);
            Assert.AreEqual(transposed[1, 1], 5);
            Assert.AreEqual(transposed[2, 0], 3);
            Assert.AreEqual(transposed[2, 1], 6);
        }

        [TestMethod]
        public void MatrixInverting()
        {
            var matrix = Matrix.Random(3, 3, 10);
            var inverted = matrix.Inverted;
            var identity = matrix * inverted;
            Assert.AreEqual(1, Math.Round(identity[0, 0]));
            Assert.AreEqual(1, Math.Round(identity[1, 1]));
            Assert.AreEqual(1, Math.Round(identity[2, 2]));
        }

        [TestMethod]
        public void MatrixMakeLU()
        {
            var m = new Matrix(3, 3)
            {
                [0, 0] = 1,
                [0, 1] = 2,
                [0, 2] = 3,
                [1, 0] = 4,
                [1, 1] = 5,
                [1, 2] = 6,
                [2, 0] = 7,
                [2, 1] = 8,
                [2, 2] = 9
            };
            //1 2 3
            //4 5 6
            //7 8 9
            var l = m.L;            
            Assert.AreEqual(0, Math.Floor(l[0, 1]));
            Assert.AreEqual(0, Math.Floor(l[0, 2]));            
            Assert.AreEqual(0, Math.Floor(l[1, 2]));
            
            var u = m.U;
            
            Assert.AreEqual(0, Math.Floor(l[1, 0]));            
            Assert.AreEqual(0, Math.Floor(l[2, 0]));
            Assert.AreEqual(0, Math.Floor(l[2, 1]));            
        }
    }
}

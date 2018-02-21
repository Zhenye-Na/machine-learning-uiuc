"""Unit Tests examples for mp6."""

from back_prop import Neural_Network
import unittest
import numpy as np


class DeepLearningTest(unittest.TestCase):
	
	def test_sigmoidPrime(self):
		NN = Neural_Network()
		s = np.array([0.2, 3.4, 7.4])
		d_sig_1 = NN.sigmoidPrime(s)
		d_sig_2 = s * (1 - s)
		self.assertEqual(d_sig_1[0],d_sig_2[0])
		self.assertEqual(d_sig_1[1],d_sig_2[1])
		self.assertEqual(d_sig_1[2],d_sig_2[2])


	def test_forward(self):
		NN = Neural_Network()
		X = np.random.randint(0, high=10, size=[3,2], dtype='l')
		X = X/np.amax(X, axis=0) 
		sol_1 = NN.forward(X)
		z = np.dot(X, NN.U) + NN.e       
		b = NN.sigmoid(z)        
		h = np.dot(b, NN.W) + NN.f      
		sol_2 = NN.sigmoid(h)
		self.assertEqual(sol_1[0][0],sol_2[0][0])
		self.assertEqual(sol_1[1][0],sol_2[1][0])
		self.assertEqual(sol_1[2][0],sol_2[2][0])


	def test_d_loss_o(self):
		NN = Neural_Network()
		gt = np.array([0.2, 3.4, 7.4])
		o = np.array([2.2, 5.4, 9.4])
		d_o_1 = NN.d_loss_o(gt, o)
		d_o_2 = np.array([2., 2., 2.])
		self.assertAlmostEqual(d_o_1[0],d_o_2[0])
		self.assertAlmostEqual(d_o_1[1],d_o_2[1])
		self.assertAlmostEqual(d_o_1[2],d_o_2[2])


if __name__ == '__main__':
	unittest.main()


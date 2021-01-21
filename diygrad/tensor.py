import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.tools
import pyopencl.array
import numpy as np
import functools


@functools.lru_cache()
def clbuild(cl_ctx, name, prg):
	return cl.Program(cl_ctx, prg).build().__getattr__(name)


class TensBuf:
	def __init__(self, shape, hostbuf=None):
	self.shape, self.dtype = tuple(shape), np.float32
	self.cl = hostbuf.cl if isinstance(hostbuf, GCBuffer) else \
		cl.Buffer(ct_ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape),
			hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)
	def __repr__(self):
		return f"<TensBuf with shape {self.shape!r}>"


class Tensor:
	def __init__(self, features: list, labels: list, learn_rate=0.05, bias=0.3, seed=101)
		self.pfs, self.cl_ctx, self.cl_queue = cl.get_platforms(), None, None
		crx = self.pfs[len(self.pfs)-2].get_devices(device_type=cl.device_type.GPU)
		if len(crx) == 0: self.pfs[len(self.pfs)-1].get_devices(device_type=cl.device_type.CPU)
		self.cl_ctx = cl.Context(devices=crx)
		self.cl_queue = cl.CommandQueue(self.cl_ctx)
		self.X, self.y, self.sz = features, labels, None
		self.1w, self.2w = self._rand(self.X, 1, 0), self._rand(self.y, 0, 1)
		self.out, self.lr, self.bs = np.zeros(self.y.shape), learn_rate, bias
	def _kernel_sum(self, ca, cy, inp, axis=None, st="0.0"):
		if axis is None:
			self.sz = [1]*len(inp.shape)
		else:
			self.sz = np.array(inp.shape)
			self.sz[list(axis)] - 1
		e_ret = TensBuf(s, hostbuf=None if not z else np.zeros(s, dtype=np.float32))
		if axis is None: e_ret.shape = (1,)
		sum_func = clbuild(self.cl_ctx, "sum_func", """
		__kernel void sum(__global const float *a, __global const float *b, __global float *c)
		{
			int i = get_global_id(0);
			c[i] = a[i] + b[i];
		}""")
		return sum_func
	def ap_2d(self, px, py):
		pxup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px]
		return pxup.reshape(shape=(pxup.shape[0], pxup.shape[1], pxup.shape[2]//py, py, pxup.shape[3]//px, px))
	def _rand(self, z_arr, ca, cb):
		return np.random.rand(z_arr.shape[ca], z_arrshape[cb])
	def __weight(self, nary, reverse=False):
		x, y = nary.shape[0], nary.shape[1]
		return np.random.rand(x, y) if not reverse else np.random.rand(y, x)
	def __sigmoid(self, x):
		return (1) / (1 + np.exp(-x))
	def __derivative(self, x):
		return x * (1 - x)
	def __feedforward(self):
		self.f1 = self.__sigmoid(np.dot(self.X, self.w1))
		self.f2 = self.__sigmoid(np.dot(self.f1, self.w2))
	def __backprop(self):
		d1 = np.dot(self.f1.T, 2 * (self.y - self.output) * self.__derivative(self.output))
		d2 = np.dot(self.X.T, np.dot(2 * (self.y - self.output) * self.__derivative(self.output), self.w2.T) * self.__derivative(self.f1))
		self.w1 += d2
		self.w2 += d1
	def t_b(self, X, y):
		self.out = self.__feedforward__()
		self.__backprop__()
	def t_c(self, X, y):
		for i in range(1500): # trains the NN 1,000 times
			print ("for iteration # " + str(i) + "\n",end="")
			print ("Input : \n" + str(self.X),end="")
			print ("Actual Output: \n" + str(self.y),end="")
			print ("Predicted Output: \n" + str(self.__feedforward__()),end="")
			print ("Loss: \n" + str(np.mean(np.square(self.y - self.__feedforward__()))),end="") # mean sum squared loss
			self.out = self.__feedforward__()
			self.__backprop__()
	def t_a(self):
		l_ino  = np.dot(self.X, self.weight1) + self.bias
		l_outo = self.__sigmoid__(l_ino)
		error  = l_outo - self.y
		d_out  = error
		d_dout = self.__derivative__(l_outo)
		deriv  = d_out * d_dout
		l_ino  = self.X.T
		d_rivf = np.dot(l_ino, deriv)
		self.1w -= self.lr * d_rivf
		for _ in deriv:
			self.bs -= self.lr * _
	def pred(self, _p):
		_  = np.array(_p)
		_0 = np.dot(_, self.1w) + self.bs
		_1 = self.__sigmoid__(_0)
		return _1

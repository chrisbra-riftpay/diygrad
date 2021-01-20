import pyopencl as cl
import pyopencl.array as pycl_array
import pyopencl.tools
import pyopencl.array
import numpy as np
import functools

pfs = cl.get_platforms()
ct_ctx, ct_queue, ct_size = None, None, len(pfs)
o_size = None

def rigc_context():
	global ct_ctx, ct_queue
	cur_ctx = pfs[ct_size-2].get_devices(device_type=cl.device_type.GPU)
	if len(cur_ctx) == 0:
		cur_ctx = pfs[ct_size-1].get_devices(device_type=cl.device_type.CPU)
	ct_ctx = cl.Context(devices=cur_ctx)
	print(f'Located -> {ct_ctx}')
	ct_queue = cl.CommandQueue(ct_ctx)

class GCBuffer:
	def __init__(self, shape, hostbuf=None):
		self.shape, self.dtype = tuple(shape), np.float32
		self.cl = hostbuf.cl if isinstance(hostbuf, GCBuffer) else \
			cl.Buffer(ct_ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape),
				hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)
	def __repr__(self):
		return f"<GPUBuffer with shape {self.shape!r}>"

def igcb(ctx, s, z=False):
	return GCBuffer(s, hostbuf=None if not z else np.zeros(s, dtype=np.float32))

@functools.lru_cache()
def clbuild(cl_ctx, name, prg):
	return cl.Program(cl_ctx, prg).build().__getattr__(name)

def kernel_sum_ops(ctx, ca, cy, inp, axis=None, st="0.0"):
	global o_size
	if axis is None: 
		o_size = [1]*len(inp.shape)
	else:
		o_size = np.array(inp.shape)
		o_size[list(axis)] - 1
	r = igcb(ct_ctx, o_size)
	if axis is None: r.shape = (1, )
	sum = clbuild(ct_ctx, "sum", """
	__kernel void sum(__global const float *a, __global const float *b, __global float *c)
	{
		int i = get_global_id(0);
		c[i] = a[i] + b[i];
	}""")
	x = pycl_array.to_device(ct_queue, np.random.rand(50000).astype(np.float32))
	y = pycl_array.to_device(ct_queue, np.random.rand(50000).astype(np.float32))
	z = pycl_array.empty_like(x)
	sum(ct_queue, x.shape, None, x.data, y.data, z.data)
	print("x: {}".format(x))
	print("y: {}".format(y))
	print("z: {}".format(z))  

class Tensor:
	def __init__(self, features: list, labels: list, learn_rate=0.05, bias=0.3, seed=101):
		self.X, self.y = features, labels
	def ap_2d(self, px, py):
		pxup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px]
		return pxup.reshape(shape=(pxup.shape[0], pxup.shape[1], pxup.shape[2]//py, py, pxup.shape[3]//px, px))

class NetworkContext:
	def __init__(self, features: list, labels: list, learn_rate=0.05, bias=0.3, seed=101):
		np.random.seed(seed)
		self.X, self.y  = features, labels
		self.w1    = np.random.rand(self.X.shape[1], self.X.shape[0])
		self.w2    = np.random.rand(self.y.shape[0], self.y.shape[1])
		self.o     = np.zeros(self.y.shape)
		self.lr    = learn_rate
		self.b     = bias

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
		self.output = self.__feedforward__()
		self.__backprop__()

	def t_c(self, X, y):
		for i in range(1500): # trains the NN 1,000 times
			print ("for iteration # " + str(i) + "\n",end="")
			print ("Input : \n" + str(self.X),end="")
			print ("Actual Output: \n" + str(self.y),end="")
			print ("Predicted Output: \n" + str(self.__feedforward__()),end="")
			print ("Loss: \n" + str(np.mean(np.square(self.y - self.__feedforward__()))),end="") # mean sum squared loss
			self.output = self.__feedforward__()
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
		self.weight1 -= self.learn_rate * d_rivf
		for _ in deriv:
			self.bias -= self.learn_rate * _

	def pred(self, _p):
		_  = np.array(_p)
		_0 = np.dot(_, self.weight1) + self.bias
		_1 = self.__sigmoid__(_0)
		return _1

def system_info():
	"""
	Device introspection
	"""
	print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
	for pf in cl.get_platforms():
		print('=' * 60)
		print('Platform - Name:  ' + pf.name)
		print('Platform - Vendor:  ' + pf.vendor)
		print('Platform - Version:  ' + pf.version)
		print('Platform - Profile:  ' + pf.profile)
		for device in pf.get_devices():
			print('-' * 60)
			print('|Device - Name:  ' + device.name)
			print('|Device - Type:  ' + cl.device_type.to_string(device.type))
			print('|Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
			print('|Device - Compute Units:  {0}'.format(device.max_compute_units))
			print('|Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024.0))
			print('|Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024.0))
			print('|Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
			print('|Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size/1048576.0))
			print('|Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
			print('-' * 60)
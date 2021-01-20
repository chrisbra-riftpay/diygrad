from diygrad import tensor

import numpy as np
# System info and available devices
# tensor.system_info()

# Example output of device selected
# Located -> <pyopencl.Context at 0x5586a52a64b0 on <pyopencl.Device 'Oclgrind Simulator' on 'Oclgrind' at 0x5586a52dd760>>

array = np.random.rand(10,3)

tensor.rigc_context()
tensor.kernel_sum_ops(None, None, None, array)
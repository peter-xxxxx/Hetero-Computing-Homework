#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class cudaModule:
    def __init__(self, idata):
        # idata: an array of lower characters.
        # TODO:
        # Declare host variables
        # Device memory allocation
        # Kernel code
        # -- initialize the device
        self.a_cpu = idata
        self.L = len(idata)

        import pycuda.autoinit

        self.kernel_code = """
        __global__ void CapCharKernel(char *a, char *b)
        {
            // 1D Thread ID (assuming that only *one* block will be executed)
            int tx = threadIdx.x;
            b[tx] = a[tx] - 32;
        }
        """

        # compile the kernel code
        mod = compiler.SourceModule(self.kernel_code)

        # get the kernel function from the compiled module
        self.capchar = mod.get_function("CapCharKernel")

        self.b_gpu = gpuarray.empty(self.L, 'S1')

    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # Memory copy to device
        # Function call and measuring time here
        # Memory copy to host
        # Return output and measured time

        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(self.a_cpu)

        start = time.time()
        # call the kernel on the card
        self.capchar(
            # inputs
            a_gpu, self.b_gpu,
            # (only one) block of L
            block = (self.L, 1, 1),
            )

        end = time.time()
        return [self.b_gpu.get(), end-start]

    def runAdd_serial(self):
        #return: an array containing capitalized characters from idata and running time.
        b_cpu = np.empty(self.L, 'S1')

        start = time.time()
        for i in range(self.L):
            b_cpu[i] = chr(ord(self.a_cpu[i])-32)

        end = time.time()
        return [b_cpu, end-start]

# generate a char array of 'a' to 'z'
alphabet = np.empty(26, 'S1')
for i in range(26):
    alphabet[i] = chr(ord('a')+i)

# concatenate sequence
a_cpu = np.tile(alphabet, 1)

# call a cudaModule object
cudamo = cudaModule(a_cpu)

# run GPU adding and CPU adding
b_gpu = cudamo.runAdd_parallel()
b_cpu = cudamo.runAdd_serial()

# show result
print 'input=\n', a_cpu
print 'py_output=\n', b_cpu # py_output is the output of your serial function
print 'parallel_output=\n', b_gpu # parallel_output is the output of your parallel function
print 'Code equality:\n', (b_cpu[0]==b_gpu[0])

# concatenate sequence
a_cpu = np.tile(alphabet, 3)

# call a cudaModule object
cudamo = cudaModule(a_cpu)

# run GPU adding and CPU adding
b_gpu = cudamo.runAdd_parallel()
b_cpu = cudamo.runAdd_serial()

# show result
print '--------------------------------'
print 'input=\n', a_cpu
print 'py_output=\n', b_cpu # py_output is the output of your serial function
print 'parallel_output=\n', b_gpu # parallel_output is the output of your parallel function
print 'Code equality:\n', (b_cpu[0]==b_gpu[0])
print '--------------------------------'


# determine the values of L when the GPU (L_CL for OpenCL and L_CUDA for CUDA)
# execution time becomes shorter than the CPU-only (Python) execution time
# record time data and plot

x_axis = []
python_time_list = []
gpu_time_list = []

for repeat_time in range(1,30):
    a_cpu = np.tile(alphabet, repeat_time)
    cudamo = cudaModule(a_cpu)

    M = 5
    times = []
    for i in range(M):
        out = cudamo.runAdd_serial()
        times.append(out[1])
    python_time_list.append(np.average(times))

    times = []
    for i in range(M):
        out = cudamo.runAdd_parallel()
        times.append(out[1])
    gpu_time_list.append(np.average(times))

    x_axis.append(repeat_time)

    print 'string_len=', len(a_cpu), '\tpy_time: ', python_time_list[-1], '\tparallel_time: ', gpu_time_list[-1]
    # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.

print '--------------------------------'
print 'python time list:\n', python_time_list
print 'gpu time list:\n', gpu_time_list

#  plot
plt.plot(x_axis, python_time_list, label='cpu')
plt.plot(x_axis, gpu_time_list, label='gpu')

plt.title("cost time vs L on cuda")

plt.xlabel('L')
plt.ylabel('time cost')

plt.legend()

plt.savefig('cuda.png')

#!/usr/bin/env python

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class openclModule:
    def __init__(self, idata):
        # idata: an array of lowercase characters.
        # Get platform and device
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # TODO:
        # Set up a command queue:
        # host variables
        # device memory allocation
        # kernel code
        self.a_cpu = idata
        self.L = len(idata)

        # Set up a command queue; we need to enable profiling to time GPU operations:
        ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        kernel = """
        __kernel void func(__global char* a, __global char* b) {
            unsigned int i = get_global_id(0);
            b[i] = a[i] - 32;
        }
        """

        self.a_gpu = cl.array.to_device(self.queue, idata)
        self.b_gpu = cl.array.empty(self.queue, idata.shape, idata.dtype)
        self.prg = cl.Program(ctx, kernel).build()


    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # function call
        # memory copy to host
        # Return output and measured time

        start = time.time()
        self.prg.func(self.queue, self.a_cpu.shape, None, self.a_gpu.data, self.b_gpu.data)
        end = time.time()

        return [self.b_gpu.get(), end-start]

    def runAdd_serial(self):
        #return: an array containing capitalized characters from idata and running time.
        
        b_cpu = np.empty_like(self.a_cpu)
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

# call a openclModule object
openclmo = openclModule(a_cpu)

# run GPU adding and CPU adding
b_gpu = openclmo.runAdd_parallel()
b_cpu = openclmo.runAdd_serial()

# show result
print 'input=\n', a_cpu
print 'py_output=\n', b_cpu # py_output is the output of your serial function
print 'parallel_output=\n', b_gpu # parallel_output is the output of your parallel function
print 'Code equality:\n', (b_cpu[0]==b_gpu[0])
print '--------------------------------'

# concatenate sequence
a_cpu = np.tile(alphabet, 3)

# call a openclModule object
openclmo = openclModule(a_cpu)

# run GPU adding and CPU adding
b_gpu = openclmo.runAdd_parallel()
b_cpu = openclmo.runAdd_serial()

# show result
print 'input=\n', a_cpu
print 'py_output=\n', b_cpu # py_output is the output of your serial function
print 'parallel_output=\n', b_gpu # parallel_output is the output of your parallel function
print 'Code equality:\n', (b_cpu[0]==b_gpu[0])
print '--------------------------------'

x_axis = []
python_time_list = []
gpu_time_list = []

for repeat_time in range(1,30):
    a_cpu = np.tile(alphabet, repeat_time)
    openclmo = openclModule(a_cpu)

    M = 10
    times = []
    for i in range(M):
        times.append(openclmo.runAdd_serial()[1])
    python_time_list.append(np.average(times))

    times = []
    for i in range(M):
        times.append(openclmo.runAdd_parallel()[1])
    gpu_time_list.append(np.average(times))

    x_axis.append(repeat_time)

    print 'string_len=', len(a_cpu), '\tpy_time: ', python_time_list[-1], '\tparallel_time: ', gpu_time_list[-1]
    # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.

print '--------------------------------'
print 'python time list:\n', python_time_list
print 'gpu time list:\n', gpu_time_list

plt.plot(x_axis, python_time_list, label='cpu')
plt.plot(x_axis, gpu_time_list, label='gpu')

plt.title("cost time vs L on OpenCL")

plt.xlabel('L')
plt.ylabel('time cost')

plt.legend()

plt.savefig('opencl.png')

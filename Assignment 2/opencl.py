#!/usr/bin/env python

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv

class Transpose:
    def __init__(self):
        # Get platform and device
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Set up a command queue; we need to enable profiling to time GPU operations:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    def transpose(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix.
        # return: the transpose of a_cpu

        M = np.int32(a_cpu.shape[0])
        N = np.int32(a_cpu.shape[1])

        block_size = 16

        kernel = """
        #define BLOCK_SIZE %(block_size)d
        #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
        #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

        __kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
        void transpose(__global float *a, __global float *a_t, int a_width, int a_height, __local float *a_local)
        {
            int base_idx_a_x   = get_group_id(0) * BLOCK_SIZE;
            int base_idx_a_y   = get_group_id(1) * BLOCK_SIZE;
            int base_idx_a_t_x = get_group_id(1) * BLOCK_SIZE;
            int base_idx_a_t_y = get_group_id(0) * BLOCK_SIZE;

            int glob_idx_a_x   = base_idx_a_x + get_local_id(0);
            int glob_idx_a_y   = base_idx_a_y + get_local_id(1);
            int glob_idx_a_t_x = base_idx_a_t_x + get_local_id(0);
            int glob_idx_a_t_y = base_idx_a_t_y + get_local_id(1);

            if (glob_idx_a_x < a_width && glob_idx_a_y < a_height)
                a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = a[glob_idx_a_x + glob_idx_a_y * a_width];
            else
                a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = 0.0;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (glob_idx_a_t_y < a_width && glob_idx_a_t_x < a_height)
                a_t[glob_idx_a_t_x + glob_idx_a_t_y * a_height] = a_local[get_local_id(0)*BLOCK_SIZE+get_local_id(1)];

        }
        """ % {"block_size": block_size}


        prg = cl.Program(self.ctx, kernel).build()

        block_x = int(np.ceil(np.float32(N)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(M)/block_size))

        start_mem = time.time()

        a_gpu = cl.array.to_device(self.queue, a_cpu)
        b_gpu = cl.array.empty(self.queue, a_cpu.T.shape, a_cpu.dtype)

        a_local = cl.LocalMemory(4*block_size*(block_size+1))

        start = time.time()
        prg.transpose(self.queue, (block_x*block_size, block_y*block_size), (block_size, block_size),
                      a_gpu.data, b_gpu.data, N, M, a_local)
        end = time.time()

        b_cpu = b_gpu.get()

        end_mem = time.time()

        if flag == 0:
            return b_cpu
        else:
            return b_cpu, end-start, end_mem-start_mem

    def transpose2(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix.
        # return: the transpose of a_cpu
        """
        : This is another parallel method
        """

        # kernel code

        M = np.int32(a_cpu.shape[0])
        N = np.int32(a_cpu.shape[1])

        block_size = 16

        kernel = """
        #define BLOCK_SIZE %(block_size)d
        #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
        #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

        __kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
        void transpose(__global float *a, __global float *a_t, int a_width, int a_height)
        {
            int base_idx_a_x   = get_group_id(0) * BLOCK_SIZE;
            int base_idx_a_y   = get_group_id(1) * BLOCK_SIZE;
            int base_idx_a_t_x = get_group_id(1) * BLOCK_SIZE;
            int base_idx_a_t_y = get_group_id(0) * BLOCK_SIZE;

            int glob_idx_a_x   = base_idx_a_x + get_local_id(0);
            int glob_idx_a_y   = base_idx_a_y + get_local_id(1);
            int glob_idx_a_t_x = base_idx_a_t_x + get_local_id(1);
            int glob_idx_a_t_y = base_idx_a_t_y + get_local_id(0);

            if (glob_idx_a_x < a_width && glob_idx_a_y < a_height)
                a_t[glob_idx_a_t_x + glob_idx_a_t_y * a_height] = a[glob_idx_a_x + glob_idx_a_y * a_width];

        }
        """ % {"block_size": block_size}


        prg = cl.Program(self.ctx, kernel).build()

        block_x = int(np.ceil(np.float32(N)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(M)/block_size))

        start_mem = time.time()

        a_gpu = cl.array.to_device(self.queue, a_cpu)
        b_gpu = cl.array.empty(self.queue, a_cpu.T.shape, a_cpu.dtype)

        start = time.time()
        prg.transpose(self.queue, (block_x*block_size, block_y*block_size), (block_size, block_size),
                      a_gpu.data, b_gpu.data, N, M)
        end = time.time()

        b_cpu = b_gpu.get()

        end_mem = time.time()

        if flag == 0:
            return b_cpu
        else:
            return b_cpu, end-start, end_mem-start_mem

    def transpose_serial(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix.
        # return: the transpose of a_cpu

        M = np.int32(a_cpu.shape[0])
        N = np.int32(a_cpu.shape[1])

        start_mem = time.time()

        b_cpu = np.empty((N, M), np.float32)

        start = time.time()

        for i in range(0, M):
            for j in range(0, N):
                b_cpu[j][i] = a_cpu[i][j]

        end = time.time()
        end_mem = time.time()

        if flag == 0:
            return b_cpu
        else:
            return b_cpu, end-start, end_mem-start_mem

class MatrixMultiply:
    def __init__(self):
        # Get platform and device
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Set up a command queue; we need to enable profiling to time GPU operations:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    def matrix_mul_naive(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix generated in cpu.
        # return: the multiplication of a_cpu and its transpose

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        block_size = 16

        kernel = """
        #define BLOCK_SIZE %(block_size)d

        __kernel void MatrixMulNaive(int m, int n, int k,
                __global float* A,  __global float* B,  __global float* C,
                __local float* ds_A, __local float* ds_B)
        {

            int Row = get_group_id(1)*BLOCK_SIZE + get_local_id(1);
            int Col = get_group_id(0)*BLOCK_SIZE + get_local_id(0);

            if ((Row < m) && (Col < k)) {
                float Cvalue = 0.0;
                for (int i = 0; i < n; ++i) {
                    Cvalue += A[Row*n+i] * B[Col+i*k];
                }
                C[Row*k+Col] = Cvalue;

            }

        }
        """ % {"block_size": block_size}


        prg = cl.Program(self.ctx, kernel).build()

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        a_gpu = cl.array.to_device(self.queue, a_cpu)
        b_gpu = cl.array.to_device(self.queue, a_cpu.T.copy())

        c_gpu = cl.array.empty(self.queue, (m, k), a_cpu.dtype)

        A_block = cl.LocalMemory(4*block_size*(block_size+1))
        B_block = cl.LocalMemory(4*block_size*(block_size+1))

        start = time.time()
        prg.MatrixMulNaive(self.queue, (block_x*block_size, block_y*block_size), (block_size, block_size),
                           m, n, k, a_gpu.data, b_gpu.data, c_gpu.data, A_block, B_block)
        end = time.time()

        c_cpu = c_gpu.get()

        end_mem = time.time()

        if flag == 0:
            return c_cpu
        else:
            return c_cpu, end-start, end_mem-start_mem

    def matrix_mul_optimized1(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix generated in cpu.
        # return: the multiplication of a_cpu and its transpose

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        block_size = 16
        tile_size = 16

        kernel = """
        #define TILE_WIDTH %(tile_size)d
        __kernel void MatrixMulSharedMem(int m, int n, int k,
                                         __global float* A,  __global float* B,  __global float* C,
                                         __local float* ds_A, __local float* ds_B)
        {
            int bx = get_group_id(0);
            int by = get_group_id(1);
            int tx = get_local_id(0);
            int ty = get_local_id(1);
            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;
            float Cvalue = 0;

            for (int t = 0; t < (n-1)/TILE_WIDTH + 1; ++t) {

                if(Row < m && t*TILE_WIDTH+tx < n)
                    ds_A[ty*TILE_WIDTH + tx] = A[Row*n + t*TILE_WIDTH+tx];
                else
                    ds_A[ty*TILE_WIDTH + tx] = 0;
                if (t*TILE_WIDTH+ty < n && Col < k)
                    ds_B[tx*TILE_WIDTH + ty] = B[(t*TILE_WIDTH+ty)*k + Col];
                else
                    ds_B[tx*TILE_WIDTH + ty] = 0;

                barrier(CLK_LOCAL_MEM_FENCE);

                #pragma unroll
                for (int i = 0; i < TILE_WIDTH; ++i)
                    Cvalue += ds_A[ty*TILE_WIDTH + i] * ds_B[tx*TILE_WIDTH + i];

                barrier(CLK_LOCAL_MEM_FENCE);

            }
            if (Row < m && Col < k)
                C[Row*k+Col] = Cvalue;

        }

        """ % {"tile_size": tile_size}


        prg = cl.Program(self.ctx, kernel).build()

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        a_gpu = cl.array.to_device(self.queue, a_cpu)
        b_gpu = cl.array.to_device(self.queue, a_cpu.T.copy())

        c_gpu = cl.array.empty(self.queue, (m, k), a_cpu.dtype)

        A_block = cl.LocalMemory(4*block_size*(block_size+1))
        B_block = cl.LocalMemory(4*block_size*(block_size+1))

        start = time.time()
        prg.MatrixMulSharedMem(self.queue, (block_x*block_size, block_y*block_size), (block_size, block_size),
                           m, n, k, a_gpu.data, b_gpu.data, c_gpu.data,
                           A_block,
                           B_block)
        end = time.time()

        c_cpu = c_gpu.get()

        end_mem = time.time()

        if flag == 0:
            return c_cpu
        else:
            return c_cpu, end-start, end_mem-start_mem

    def matrix_mul_optimized2(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix generated in cpu.
        # return: the multiplication of a_cpu and its transpose

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        block_size = 16
        tile_size = 16

        kernel = """
        #define TILE_WIDTH %(tile_size)d
        __kernel void MatrixMulOptm(int m, int n, int k,
                                         __global float* A,  __global float* B,  __global float* C,
                                         __local float* ds_A, __local float* ds_B)
        {
            int bx = get_group_id(0);
            int by = get_group_id(1);
            int tx = get_local_id(0);
            int ty = get_local_id(1);
            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;
            float Cvalue = 0;

            for (int t = 0; t < (n-1)/TILE_WIDTH + 1; ++t) {

                if(Row < m && t*TILE_WIDTH+tx < n)
                    ds_A[ty*TILE_WIDTH + tx] = A[Row*n + t*TILE_WIDTH+tx];
                else
                    ds_A[ty*TILE_WIDTH + tx] = 0;
                if (t*TILE_WIDTH+ty < n && Col < k)
                    ds_B[ty*TILE_WIDTH + tx] = B[(t*TILE_WIDTH+ty)*k + Col];
                else
                    ds_B[ty*TILE_WIDTH + tx] = 0;

                barrier(CLK_LOCAL_MEM_FENCE);

                for (int i = 0; i < TILE_WIDTH; ++i)
                    Cvalue += ds_A[ty*TILE_WIDTH + i] * ds_B[i*TILE_WIDTH + tx];

                barrier(CLK_LOCAL_MEM_FENCE);

            }
            if (Row < m && Col < k)
                C[Row*k+Col] = Cvalue;

        }

        """ % {"tile_size": tile_size}


        prg = cl.Program(self.ctx, kernel).build()

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        a_gpu = cl.array.to_device(self.queue, a_cpu)
        b_gpu = cl.array.to_device(self.queue, a_cpu.T.copy())

        c_gpu = cl.array.empty(self.queue, (m, k), a_cpu.dtype)

        A_block = cl.LocalMemory(4*block_size*(block_size+1))
        B_block = cl.LocalMemory(4*block_size*(block_size+1))

        start = time.time()
        prg.MatrixMulOptm(self.queue, (block_x*block_size, block_y*block_size), (block_size, block_size),
                           m, n, k, a_gpu.data, b_gpu.data, c_gpu.data,
                           A_block,
                           B_block)
        end = time.time()

        c_cpu = c_gpu.get()

        end_mem = time.time()

        if flag == 0:
            return c_cpu
        else:
            return c_cpu, end-start, end_mem-start_mem


def mean_with_reject_outliers(data, m=1):
    data = np.array(data)
    return np.average(data[abs(data - np.mean(data)) < m * np.std(data)])

"""
: Transpose
"""
print('====== transpose =====')
trans = Transpose()

M = 7
N = 9

a_cpu = np.random.randn(M, N).astype(np.float32)
b_cpu = trans.transpose(a_cpu)
print("===================================")
print(b_cpu)
print("===================================")

record_dict_list = []
x_axis = []
y_time_parallel = []
y_time_parallel_mem = []
y_time_parallel2 = []
y_time_parallel_mem2 = []
y_time_serial = []

for i in range(1, 10):
    time_list = []
    time_mem_list = []
    time_list2 = []
    time_mem_list2 = []
    time_serial_list = []
    for k in range(0, 5):
        a_cpu = np.random.randn(M*i, N*i).astype(np.float32)

        b_cpu, tim, tim_mem = trans.transpose(a_cpu, 1)
        print(np.array_equal(a_cpu.T, b_cpu), a_cpu.shape, b_cpu.shape)
        time_list.append(tim)
        time_mem_list.append(tim_mem)

        b_cpu, tim, tim_mem = trans.transpose2(a_cpu, 1)
        print(np.array_equal(a_cpu.T, b_cpu), a_cpu.shape, b_cpu.shape)
        time_list2.append(tim)
        time_mem_list2.append(tim_mem)

        b_cpu, tim, tim_mem = trans.transpose_serial(a_cpu, 1)
        print(np.array_equal(a_cpu.T, b_cpu), a_cpu.shape, b_cpu.shape)
        time_serial_list.append(tim)

    record_dict_list.append({'i': i,
                             'time': mean_with_reject_outliers(time_list),
                             'time include mem': mean_with_reject_outliers(time_mem_list),
                             'time2': mean_with_reject_outliers(time_list),
                             'time2 include mem': mean_with_reject_outliers(time_mem_list),
                             'time serial': mean_with_reject_outliers(time_serial_list)})

    x_axis.append(M*i*N*i)
    y_time_parallel.append(mean_with_reject_outliers(time_list))
    y_time_parallel_mem.append(mean_with_reject_outliers(time_mem_list))
    y_time_parallel2.append(mean_with_reject_outliers(time_list2))
    y_time_parallel_mem2.append(mean_with_reject_outliers(time_mem_list2))
    y_time_serial.append(mean_with_reject_outliers(time_serial_list))

with open('opencl_transpose.csv', 'w') as f:
    w = csv.DictWriter(f, record_dict_list[0].keys())
    w.writeheader()
    for dic in record_dict_list:
        w.writerow(dic)

plt.figure()

plt.plot(x_axis, y_time_serial, label='cpu')
plt.plot(x_axis, y_time_parallel, label='gpu')
plt.plot(x_axis, y_time_parallel2, label='gpu method2')

plt.title("cost time vs size on opencl")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('opencl_transpose.png')

plt.figure()

plt.plot(x_axis, y_time_serial, label='cpu')
plt.plot(x_axis, y_time_parallel_mem, label='gpu with mem copy time')
plt.plot(x_axis, y_time_parallel_mem2, label='gpu method2')

plt.title("cost time vs size on opencl include mem copy")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('opencl_transpose_mem.png')

plt.figure()

"""
: Matrix Multiply
"""
print('====== matrix multiply =====')
mul = MatrixMultiply()

M = 9
N = 8

a_cpu = np.random.randn(M, N).astype(np.float32)
c_cpu = mul.matrix_mul_naive(a_cpu)
print("===================================")
print(c_cpu)
print("===================================")

record_dict_list = []
x_axis = []
y_time_parallel = []
y_time_parallel_mem = []
y_time_parallel1 = []
y_time_parallel_mem1 = []
y_time_parallel2 = []
y_time_parallel_mem2 = []

for i in range(1, 30):
    time_list = []
    time_mem_list = []
    time_list1 = []
    time_mem_list1 = []
    time_list2 = []
    time_mem_list2 = []

    for k in range(0,10):
        a_cpu = np.random.randn(M*i, N*i).astype(np.float32)

        c_cpu, tim, tim_mem = mul.matrix_mul_naive(a_cpu, 1)
        time_list.append(tim)
        time_mem_list.append(tim_mem)
        print(np.allclose(np.dot(a_cpu,a_cpu.T), c_cpu, atol=1e-04), a_cpu.shape, c_cpu.shape)

        c_cpu, tim, tim_mem = mul.matrix_mul_optimized1(a_cpu, 1)
        time_list1.append(tim)
        time_mem_list1.append(tim_mem)
        print(np.allclose(np.dot(a_cpu,a_cpu.T), c_cpu, atol=1e-04), a_cpu.shape, c_cpu.shape)

        c_cpu, tim, tim_mem = mul.matrix_mul_optimized2(a_cpu, 1)
        time_list2.append(tim)
        time_mem_list2.append(tim_mem)
        print(np.allclose(np.dot(a_cpu,a_cpu.T), c_cpu, atol=1e-04), a_cpu.shape, c_cpu.shape)

    record_dict_list.append({'i': i,
                             'time': mean_with_reject_outliers(time_list),
                             'time include mem': mean_with_reject_outliers(time_mem_list),
                             'time optm1': mean_with_reject_outliers(time_list1),
                             'time optm1 include mem': mean_with_reject_outliers(time_mem_list1),
                             'time optm2': mean_with_reject_outliers(time_list2),
                             'time optm2 include mem': mean_with_reject_outliers(time_mem_list2)})

    x_axis.append(M*i*N*i)
    y_time_parallel.append(mean_with_reject_outliers(time_list))
    y_time_parallel_mem.append(mean_with_reject_outliers(time_mem_list))
    y_time_parallel1.append(mean_with_reject_outliers(time_list1))
    y_time_parallel_mem1.append(mean_with_reject_outliers(time_mem_list1))
    y_time_parallel2.append(mean_with_reject_outliers(time_list2))
    y_time_parallel_mem2.append(mean_with_reject_outliers(time_mem_list2))

with open('opencl_mul.csv', 'w') as f:
    w = csv.DictWriter(f, record_dict_list[0].keys())
    w.writeheader()
    for dic in record_dict_list:
        w.writerow(dic)

plt.figure()

plt.plot(x_axis, y_time_parallel, label='naive')
plt.plot(x_axis, y_time_parallel1, label='shared mem')
plt.plot(x_axis, y_time_parallel2, label='optimized2')

plt.title("cost time vs size on opencl")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('opencl_matrix_mul.png')

plt.figure()

plt.plot(x_axis, y_time_parallel_mem, label='naive')
plt.plot(x_axis, y_time_parallel_mem1, label='shared mem')
plt.plot(x_axis, y_time_parallel_mem2, label='optimized2')

plt.title("cost time vs size on opencl include memcopy")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('opencl_matrix_mul_mem.png')

plt.figure()

print("===== finish ======")

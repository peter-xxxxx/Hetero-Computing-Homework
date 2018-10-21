#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv


class Transpose:
    def transpose(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix.
        # return: the transpose of a_cpu

        import pycuda.autoinit

        block_size = 16

        kernel_code = """
         #define BLOCK_SIZE %(block_size)d

         __global__ void transpose(float *A_t, float *A, int a_width, int a_height)
         {
             // Base indices in A and A_t
             int base_idx_a_x   = blockIdx.x * BLOCK_SIZE;
             int base_idx_a_y   = blockIdx.y * BLOCK_SIZE;
             int base_idx_a_t_x = blockIdx.y * BLOCK_SIZE;
             int base_idx_a_t_y = blockIdx.x * BLOCK_SIZE;

             // Global indices in A and A_t
             int glob_idx_a_x   = base_idx_a_x + threadIdx.x;
             int glob_idx_a_y   = base_idx_a_y + threadIdx.y;
             int glob_idx_a_t_x = base_idx_a_t_x + threadIdx.x;
             int glob_idx_a_t_y = base_idx_a_t_y + threadIdx.y;

             __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE+1];

             // Store transposed submatrix to shared memory
             if (glob_idx_a_x < a_width && glob_idx_a_y < a_height)
                A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a_x + glob_idx_a_y * a_width];
             else
                A_shared[threadIdx.y][threadIdx.x] = 0;

             __syncthreads();

             // Write transposed submatrix to global memory
             if (glob_idx_a_t_y < a_width && glob_idx_a_t_x < a_height)
                A_t[glob_idx_a_t_x + glob_idx_a_t_y * a_height] = A_shared[threadIdx.x][threadIdx.y];

         }
        """ % {"block_size": block_size}

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        transpose = mod.get_function("transpose")

        M = np.int32(a_cpu.shape[0])
        N = np.int32(a_cpu.shape[1])

        block_x = int(np.ceil(np.float32(N)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(M)/block_size))

        start_mem = time.time()

        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)

        b_gpu = gpuarray.empty((N, M), np.float32)

        start = time.time()
        # call the kernel on the card
        transpose(
            b_gpu,
            a_gpu,
            N,
            M,
            block = (block_size, block_size, 1),
            grid = (block_x, block_y, 1)
        )

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

        import pycuda.autoinit

        block_size = 16

        kernel_code = """
         #define BLOCK_SIZE %(block_size)d

         __global__ void transpose2(float *A_t, float *A, int a_width, int a_height)
         {
             // Base indices in A and A_t
             int base_idx_a_x   = blockIdx.x * BLOCK_SIZE;
             int base_idx_a_y   = blockIdx.y * BLOCK_SIZE;
             int base_idx_a_t_x = blockIdx.y * BLOCK_SIZE;
             int base_idx_a_t_y = blockIdx.x * BLOCK_SIZE;

             // Global indices in A and A_t
             int glob_idx_a_x   = base_idx_a_x + threadIdx.x;
             int glob_idx_a_y   = base_idx_a_y + threadIdx.y;
             int glob_idx_a_t_x = base_idx_a_t_x + threadIdx.y;
             int glob_idx_a_t_y = base_idx_a_t_y + threadIdx.x;

             if (glob_idx_a_x < a_width && glob_idx_a_y < a_height)
                 A_t[glob_idx_a_t_x + glob_idx_a_t_y * a_height] =
                     A[glob_idx_a_x + glob_idx_a_y * a_width];

         }
        """ % {"block_size": block_size}

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        transpose2 = mod.get_function("transpose2")

        M = np.int32(a_cpu.shape[0])
        N = np.int32(a_cpu.shape[1])

        block_x = int(np.ceil(np.float32(N)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(M)/block_size))

        start_mem = time.time()

        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)

        b_gpu = gpuarray.empty((N, M), np.float32)

        start = time.time()
        # call the kernel on the card
        transpose2(
            b_gpu,
            a_gpu,
            N,
            M,
            block = (block_size, block_size, 1),
            grid = (block_x, block_y, 1)
        )

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
    def matrix_mul_naive(self, a_cpu, flag=0):
        # a_cpu: a 2D matrix generated in cpu.
        # return: the multiplication of a_cpu and its transpose

        import pycuda.autoinit

        block_size = 16

        kernel_code = """
        #define BLOCK_SIZE %(block_size)d

        __global__ void MatrixMulNaive(int m, int n, int k, float* A, float* B, float* C)
        {

            int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
            int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

            if ((Row < m) && (Col < k)) {
                float Cvalue = 0.0;
                for (int i = 0; i < n; ++i) {
                    Cvalue += A[Row*n+i] * B[Col+i*k];
                }
                C[Row*k+Col] = Cvalue;

            }
        }

        """ % {"block_size": block_size}

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        matrixMulNaive = mod.get_function("MatrixMulNaive")

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        c_gpu = gpuarray.empty((m, k), np.float32)

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        # transfer host (CPU) memory to device (GPU) memory
        b_cpu = np.transpose(a_cpu).copy()
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        start = time.time()
        # call the kernel on the card
        matrixMulNaive(
            m,
            n,
            k,
            a_gpu,
            b_gpu,
            c_gpu,
            block = (block_size, block_size, 1),
            grid = (block_x, block_y, 1)
        )

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

        import pycuda.autoinit

        block_size = 16
        tile_size = 16

        kernel_code = """

        #define TILE_WIDTH %(tile_size)d
        __global__ void MatrixMulSharedMem(int m, int n, int k, float* A, float* B, float* C)
        {

            __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * blockDim.y + ty;
            int Col = bx * blockDim.x + tx;
            float Cvalue = 0;

            for (int t = 0; t < (n-1)/TILE_WIDTH + 1; ++t) {

                if(Row < m && t*TILE_WIDTH+tx < n)
                    ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
                else
                    ds_A[ty][tx] = 0;
                if (t*TILE_WIDTH+ty < n && Col < k)
                    ds_B[tx][ty] = B[(t*TILE_WIDTH+ty)*k + Col];
                else
                    ds_B[tx][ty] = 0;

                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; ++i)
                    Cvalue += ds_A[ty][i] * ds_B[tx][i];

                __syncthreads();

            }
            if (Row < m && Col < k)
                C[Row*k+Col] = Cvalue;

        }

        """ % {"tile_size": tile_size}

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        matrixMulSharedMem = mod.get_function("MatrixMulSharedMem")

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        c_gpu = gpuarray.empty((m, k), np.float32)

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        # transfer host (CPU) memory to device (GPU) memory
        b_cpu = np.transpose(a_cpu).copy()
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        start = time.time()
        # call the kernel on the card
        matrixMulSharedMem(
            m,
            n,
            k,
            a_gpu,
            b_gpu,
            c_gpu,
            block = (block_size, block_size, 1),
            grid = (block_x, block_y, 1)
        )

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

        import pycuda.autoinit

        block_size = 16
        tile_size = 16

        kernel_code = """

        #define TILE_WIDTH %(tile_size)d
        __global__ void MatrixMulOptm(int m, int n, int k, float* A, float* B, float* C)
        {

            __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * blockDim.y + ty;
            int Col = bx * blockDim.x + tx;
            float Cvalue = 0;

            for (int t = 0; t < (n-1)/TILE_WIDTH + 1; ++t) {

                if(Row < m && t*TILE_WIDTH+tx < n)
                    ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
                else
                    ds_A[ty][tx] = 0;
                if (t*TILE_WIDTH+ty < n && Col < k)
                    ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + Col];
                else
                    ds_B[ty][tx] = 0;

                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; ++i)
                    Cvalue += ds_A[ty][i] * ds_B[i][tx];
                __syncthreads();
            }
            if (Row < m && Col < k)
                C[Row*k+Col] = Cvalue;

        }

        """ % {"tile_size": tile_size}

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        matrixMulOptm = mod.get_function("MatrixMulOptm")

        m = np.int32(a_cpu.shape[0])
        n = np.int32(a_cpu.shape[1])

        k = m

        c_gpu = gpuarray.empty((m, k), np.float32)

        block_x = int(np.ceil(np.float32(m)/block_size)) # number of blocks along X
        block_y = int(np.ceil(np.float32(k)/block_size))

        start_mem = time.time()

        # transfer host (CPU) memory to device (GPU) memory
        b_cpu = np.transpose(a_cpu).copy()
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        start = time.time()
        # call the kernel on the card
        matrixMulOptm(
            m,
            n,
            k,
            a_gpu,
            b_gpu,
            c_gpu,
            block = (block_size, block_size, 1),
            grid = (block_x, block_y, 1)
        )

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

with open('cuda_transpose.csv', 'w') as f:
    w = csv.DictWriter(f, record_dict_list[0].keys())
    w.writeheader()
    for dic in record_dict_list:
        w.writerow(dic)

plt.figure()

plt.plot(x_axis, y_time_serial, label='cpu')
plt.plot(x_axis, y_time_parallel, label='gpu')
plt.plot(x_axis, y_time_parallel2, label='gpu method2')

plt.title("cost time vs size on cuda")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('cuda_transpose.png')

plt.figure()

plt.plot(x_axis, y_time_serial, label='cpu')
plt.plot(x_axis, y_time_parallel_mem, label='gpu with mem copy time')
plt.plot(x_axis, y_time_parallel_mem2, label='gpu method2')

plt.title("cost time vs size on cuda include mem copy")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('cuda_transpose_mem.png')

plt.figure()

"""
: Matrix Multiply
"""
print('====== matrix multiply =====')
mul = MatrixMultiply()

M = 7
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

with open('cuda_mul.csv', 'w') as f:
    w = csv.DictWriter(f, record_dict_list[0].keys())
    w.writeheader()
    for dic in record_dict_list:
        w.writerow(dic)

plt.figure()

plt.plot(x_axis, y_time_parallel, label='naive')
plt.plot(x_axis, y_time_parallel1, label='shared mem')
plt.plot(x_axis, y_time_parallel2, label='optimized2')

plt.title("cost time vs size on cuda")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('cuda_matrix_mul.png')

plt.figure()

plt.plot(x_axis, y_time_parallel_mem, label='naive')
plt.plot(x_axis, y_time_parallel_mem1, label='shared mem')
plt.plot(x_axis, y_time_parallel_mem2, label='optimized2')

plt.title("cost time vs size on cuda include memcopy")
plt.xlabel('size')
plt.ylabel('time cost')
plt.legend()
plt.savefig('cuda_matrix_mul_mem.png')

plt.figure()

print("===== finish ======")

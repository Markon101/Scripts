#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <random>

// Constants for the Adam optimizer
const double beta1 = 0.9;
const double beta2 = 0.999;
const double epsilon = 1e-8;
const int max_iterations = 6000;
const double learning_rate = 0.001;
const double tolerance = 1e-5;

// CUDA Kernel for gradient descent with Adam optimizer
__global__ void gradientDescentKernel(double* x_values, double* results, double* target_values, int num_elements, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState state;
        curand_init(55412, idx, 0, &state); // Initialize random state

        // Randomize initial x within a range
        double x = x_values[idx] + (curand_uniform(&state) - 0.5) * 2.0;
        double m = 0, v = 0;
        double target = target_values[idx];

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double fx = sin(x) + 5 * pow(cos(x), 2);
            double gradient = cos(x) - 10 * cos(x) * sin(x);

            // Check if the current solution is within the tolerance of the target
            if (fabs(fx - target) < tolerance) {
                results[idx] = x; // Store the optimized value of x
                return;
            }

            // Update x using Adam optimizer
            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * gradient * gradient;
            double m_hat = m / (1 - pow(beta1, iteration + 1));
            double v_hat = v / (1 - pow(beta2, iteration + 1));
            x -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }

        results[idx] = NAN; // Indicate no solution found within tolerance
    }
}

// Function to initialize x_values for each thread
void initializeXValues(double* x_values, int num_elements, double min_x, double max_x) {
    double step = (max_x - min_x) / num_elements;
    for (int i = 0; i < num_elements; ++i) {
        x_values[i] = min_x + step * i;
    }
}

int main() {
    const int num_threads = 3200;
    const double min_x = -2000; // Define the range for x
    const double max_x = 2000;

    // Allocate and initialize arrays on the host
    double* x_values_host = new double[num_threads];
    double* results_host = new double[num_threads];
    double* target_values_host = new double[num_threads];

    // Initialize target values (example values, modify as needed)
    for (int i = 0; i < num_threads; ++i) {
        target_values_host[i] = 3.12362367; // Example target value
    }

    initializeXValues(x_values_host, num_threads, min_x, max_x);

    // Allocate memory on the device
    double* x_values_dev = nullptr;
    double* results_dev = nullptr;
    double* target_values_dev = nullptr;
    cudaMalloc(&x_values_dev, num_threads * sizeof(double));
    cudaMalloc(&results_dev, num_threads * sizeof(double));
    cudaMalloc(&target_values_dev, num_threads * sizeof(double));

    // Copy data to the device
    cudaMemcpy(x_values_dev, x_values_host, num_threads * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(target_values_dev, target_values_host, num_threads * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block sizes for kernel launch
    int blockSize = 1024; // Optimize this value based on your hardware
    int numBlocks = (num_threads + blockSize - 1) / blockSize;

    // Launch the kernel with target values
    gradientDescentKernel<<<numBlocks, blockSize>>>(x_values_dev, results_dev, target_values_dev, num_threads, learning_rate);
    cudaDeviceSynchronize(); // Wait for the kernel to complete

    // Check for errors during launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        // Handle error...
    }

    // Copy results back to the host
    cudaMemcpy(results_host, results_dev, num_threads * sizeof(double), cudaMemcpyDeviceToHost);

    // Output the results close to target values
    for (int i = 0; i < num_threads; ++i) {
        if (!isnan(results_host[i])) {
            std::cout << "Result for target = " << target_values_host[i] << ": x ≈ " << results_host[i] << std::endl;
        }
    }

    // Clean up
    cudaFree(x_values_dev);
    cudaFree(results_dev);
    cudaFree(target_values_dev);
    delete[] x_values_host;
    delete[] results_host;
    delete[] target_values_host;

    return 0;
}


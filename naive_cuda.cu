#include <iostream>   
#include <vector>     
#include <cuda_runtime.h>
#include <iomanip>    

// CUDA 오류 체크 매크로
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " code: " << err << " (" << cudaGetErrorString(err)      \
                      << ")" << std::endl;                                        \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Naive CUDA Kernel: 각 스레드가 결과 행렬 C의 한 요소를 계산
__global__ void matrixMultiply_Naive_GPU(const float* A,
                                         const float* B,
                                         float* C,
                                         int rowsA,
                                         int colsA,
                                         int colsB)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // C의 열 인덱스
    int row = blockIdx.y * blockDim.y + threadIdx.y; // C의 행 인덱스

    // C 행렬의 유효 범위 내에 있는지 확인
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// 특정 크기의 행렬에 대해 GPU 성능을 측정하는 함수
double benchmarkMatrixMultiplication(int size) {
    int N = size; // A의 행 수
    int M = size; // A의 열 수 (B의 행 수)
    int K = size; // B의 열 수

    std::cout << "행렬 크기 " << N << "x" << K << " 측정 중..." << std::endl;

    // 호스트(CPU) 메모리에 행렬 할당
    std::vector<float> h_A(N * M);
    std::vector<float> h_B(M * K);
    std::vector<float> h_C(N * K);

    // 행렬 초기화 (무작위 값)
    for (int i = 0; i < N * M; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < M * K; ++i) {
        h_B[i] = static_cast<float>(rand() % 100) / 10.0f;
    }

    // 디바이스(GPU) 메모리 할당
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * K * sizeof(float)));

    // 호스트에서 디바이스로 데이터 복사
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA 이벤트 생성 (GPU 타이밍)
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // 블록 및 그리드 차원 설정
    int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    int num_runs_gpu = 20; // 20회 실행
    double total_execution_time_gpu_ms = 0.0;

    for (int run = 0; run < num_runs_gpu; ++run) {
        // 결과 행렬 C 초기화
        CUDA_CHECK(cudaMemset(d_C, 0, N * K * sizeof(float))); 

        CUDA_CHECK(cudaEventRecord(start_gpu));
        matrixMultiply_Naive_GPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M, K);
        CUDA_CHECK(cudaEventRecord(stop_gpu));
        CUDA_CHECK(cudaEventSynchronize(stop_gpu));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        total_execution_time_gpu_ms += milliseconds;
    }
    
    double avg_gpu_time_ms = total_execution_time_gpu_ms / num_runs_gpu;

    // 디바이스 메모리 해제
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // CUDA 이벤트 해제
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    return avg_gpu_time_ms;
}

int main() {
    // 테스트할 행렬 크기 배열
    std::vector<int> matrix_sizes = {16, 32, 64, 128, 512, 1024};
    
    std::cout << "===== GPU 기반 행렬 곱셈 성능 측정 =====" << std::endl;
    std::cout << std::left << std::setw(12) << "행렬 크기" << std::setw(20) << "평균 실행 시간(ms)" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    // 각 크기에 대해 측정 실행
    for (int size : matrix_sizes) {
        double avg_time = benchmarkMatrixMultiplication(size);
        std::cout << std::left << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size)) 
                  << std::setw(20) << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }
    
    return 0;
}
#include <iostream>
#include <vector>
#include <chrono>
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

// Tiled CUDA Kernel: 공유 메모리를 사용한 타일 기반 행렬 곱셈
template<int TILE_SIZE>
__global__ void matrixMultiply_Tiled_GPU(const float* A,
                                         const float* B,
                                         float* C,
                                         int rowsA,
                                         int colsA,
                                         int colsB)
{
    // 공유 메모리 동적 할당 (커널 실행 구성에서 크기 지정)
    extern __shared__ float shared_mem[];
    
    // 공유 메모리를 두 부분으로 분할
    float* tile_A = shared_mem;                    // A의 타일
    float* tile_B = shared_mem + TILE_SIZE * TILE_SIZE; // B의 타일
    
    int bx = blockIdx.x;  // 블록의 x 인덱스
    int by = blockIdx.y;  // 블록의 y 인덱스
    int tx = threadIdx.x; // 스레드의 x 인덱스
    int ty = threadIdx.y; // 스레드의 y 인덱스
    
    // C 행렬에서 현재 스레드가 계산할 요소의 글로벌 인덱스
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 타일 단위로 행렬 곱셈 수행
    for (int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // A의 타일 로드 (경계 검사 포함)
        if (row < rowsA && t * TILE_SIZE + tx < colsA) {
            tile_A[ty * TILE_SIZE + tx] = A[row * colsA + (t * TILE_SIZE + tx)];
        } else {
            tile_A[ty * TILE_SIZE + tx] = 0.0f;
        }
        
        // B의 타일 로드 (경계 검사 포함)
        if (t * TILE_SIZE + ty < colsA && col < colsB) {
            tile_B[ty * TILE_SIZE + tx] = B[(t * TILE_SIZE + ty) * colsB + col];
        } else {
            tile_B[ty * TILE_SIZE + tx] = 0.0f;
        }
        
        // 모든 스레드가 공유 메모리 로드를 완료할 때까지 대기
        __syncthreads();
        
        // 현재 타일을 사용하여 부분합 계산
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty * TILE_SIZE + k] * tile_B[k * TILE_SIZE + tx];
        }
        
        // 다음 타일을 로드하기 전에 동기화
        __syncthreads();
    }
    
    // 결과를 C 행렬에 저장 (경계 검사 포함)
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

// CPU 기반 직렬 행렬 곱셈 함수 (정확성 검증용)
void matrixMultiply_CPU(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int rowsA,
                        int colsA,
                        int colsB)
{
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// 특정 크기의 행렬에 대해 CPU와 GPU 성능을 측정하는 함수
void benchmarkMatrixMultiplication(int size) {
    int N = size; // A의 행 수
    int M = size; // A의 열 수 (B의 행 수)
    int K = size; // B의 열 수

    std::cout << "\n==================================================" << std::endl;
    std::cout << "행렬 크기: " << N << "x" << M << " * " << M << "x" << K << std::endl;
    std::cout << "==================================================" << std::endl;

    // 호스트(CPU) 메모리에 행렬 할당
    std::vector<float> h_A(N * M);
    std::vector<float> h_B(M * K);
    std::vector<float> h_C_cpu(N * K);   // CPU 결과 저장용
    std::vector<float> h_C_gpu(N * K);   // GPU 결과 저장용

    // 행렬 초기화 (무작위 값 또는 특정 패턴)
    for (int i = 0; i < N * M; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < M * K; ++i) {
        h_B[i] = static_cast<float>(rand() % 100) / 10.0f;
    }

    // --- CPU Matrix Multiplication ---
    std::cout << "CPU 행렬 곱셈 시작..." << std::endl;
    int num_runs_cpu = 20; // 최소 20회 실행
    double total_execution_time_cpu_ms = 0.0;
    
    for (int run = 0; run < num_runs_cpu; ++run) {
        std::fill(h_C_cpu.begin(), h_C_cpu.end(), 0.0f); // 매 실행마다 초기화
        auto start_cpu = std::chrono::high_resolution_clock::now(); // CPU 타이밍
        matrixMultiply_CPU(h_A, h_B, h_C_cpu, N, M, K);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_execution_time_cpu_ms += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }
    double avg_cpu_time_ms = total_execution_time_cpu_ms / num_runs_cpu;
    std::cout << "CPU 평균 실행 시간 (" << num_runs_cpu << "회): " << std::fixed << std::setprecision(3) << avg_cpu_time_ms << " ms" << std::endl;

    // --- Tiled CUDA Kernel Matrix Multiplication ---
    std::cout << "GPU 타일드 행렬 곱셈 시작..." << std::endl;

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

    // 타일 크기 설정
    const int TILE_SIZE = 16; // 16x16 타일
    
    // 블록 및 그리드 차원 설정
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // 공유 메모리 크기 계산 (A와 B의 타일을 위한 공간)
    int sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    int num_runs_gpu = 20; // 최소 20회 실행
    double total_execution_time_gpu_ms = 0.0;

    for (int run = 0; run < num_runs_gpu; ++run) {
        // 결과 행렬 C 초기화
        CUDA_CHECK(cudaMemset(d_C, 0, N * K * sizeof(float))); 

        CUDA_CHECK(cudaEventRecord(start_gpu)); // GPU 타이밍 시작
        // 공유 메모리 크기를 세 번째 매개변수로 전달
        matrixMultiply_Tiled_GPU<TILE_SIZE><<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C, N, M, K);
        CUDA_CHECK(cudaEventRecord(stop_gpu));  // GPU 타이밍 종료
        CUDA_CHECK(cudaEventSynchronize(stop_gpu)); // 커널 완료 대기

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        total_execution_time_gpu_ms += milliseconds;
    }
    double avg_gpu_time_ms = total_execution_time_gpu_ms / num_runs_gpu;
    std::cout << "GPU 평균 실행 시간 (" << num_runs_gpu << "회): " << std::fixed << std::setprecision(3) << avg_gpu_time_ms << " ms" << std::endl;
    std::cout << "속도 향상: " << std::fixed << std::setprecision(2) << (avg_cpu_time_ms / avg_gpu_time_ms) << "x" << std::endl;

    // 디바이스에서 호스트로 결과 복사
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost));

    // --- 정확성 검증 (CPU 결과와 GPU 결과 비교) ---
    bool correct = true;
    float epsilon = 1e-2f; // 더 큰 오차 허용
    int errors = 0;
    for (int i = 0; i < N * K; ++i) {
        // 상대 오차를 사용하여 비교
        float rel_error = std::abs(h_C_cpu[i] - h_C_gpu[i]) / std::max(std::abs(h_C_cpu[i]), 1e-5f);
        if (rel_error > epsilon) {
            correct = false;
            errors++;
            if (errors < 10) { // 처음 10개 에러만 출력
                std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i] << ", GPU=" << h_C_gpu[i] 
                          << ", 상대 오차=" << rel_error << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "결과 검증: 성공 (허용 오차 내 일치)" << std::endl;
    } else {
        std::cout << "결과 검증: 실패 (오류 " << errors << "개)" << std::endl;
    }

    // 디바이스 메모리 해제
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // CUDA 이벤트 해제
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
}

int main() {
    // 테스트할 행렬 크기 배열
    std::vector<int> matrix_sizes = {16, 32, 64, 128, 512, 1024};
    
    std::cout << "===== 타일링 기법을 적용한 GPU 행렬 곱셈 성능 측정 =====" << std::endl;
    
    // 결과 테이블 헤더
    std::cout << std::left << std::setw(12) << "행렬 크기" 
              << std::setw(16) << "CPU 시간(ms)" 
              << std::setw(16) << "GPU 시간(ms)"
              << std::setw(12) << "속도 향상" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    
    // 각 크기에 대해 측정 실행
    for (int size : matrix_sizes) {
        benchmarkMatrixMultiplication(size);
    }
    
    return 0;
}
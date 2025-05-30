#include <vector>     // std::vector를 사용하기 위해 포함
#include <iostream>   // 입출력을 위해 포함
#include <chrono>     // 시간 측정을 위해 포함
#include <numeric>    // std::iota (선택 사항, 행렬 초기화에 사용될 수 있음)
#include <iomanip>    // 출력 형식 지정을 위해 포함
#include <string>     // std::to_string 사용을 위해 포함

// CPU 기반 직렬 행렬 곱셈 함수
// C = A * B
void matrixMultiply_CPU(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int rowsA,    // 행렬 A의 행 수
                        int colsA,    // 행렬 A의 열 수 (행렬 B의 행 수와 같아야 함)
                        int colsB)    // 행렬 B의 열 수
{
    // C는 rowsA * colsB 크기로 이미 할당되어 있어야 합니다.

    for (int i = 0; i < rowsA; ++i) { // A 행렬의 행을 순회 (C의 행)
        for (int j = 0; j < colsB; ++j) { // B 행렬의 열을 순회 (C의 열)
            float sum = 0.0f; // 현재 계산할 C[i][j] 요소의 합계 초기화
            for (int k = 0; k < colsA; ++k) { // A의 열과 B의 행을 순회
                sum += A[i * colsA + k] * B[k * colsB + j]; // 행렬 곱셈의 핵심 연산
            }
            C[i * colsB + j] = sum; // 계산된 값을 C 행렬에 저장
        }
    }
}

// 주어진 크기의 행렬에 대해 행렬 곱셈 시간을 측정하는 함수
double measureMatrixMultiplication(int size) {
    int N = size; // 행렬 A의 행 수
    int M = size; // 행렬 A의 열 수이자 행렬 B의 행 수
    int K = size; // 행렬 B의 열 수

    // 행렬 A (N x M)
    std::vector<float> A(N * M);
    // 행렬 B (M x K)
    std::vector<float> B(M * K);
    // 결과 행렬 C (N x K)
    std::vector<float> C(N * K);

    // 행렬 초기화
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i * M + j] = static_cast<float>(i + j + 0.5f);
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i * K + j] = static_cast<float>(i * 0.1f + j * 0.01f + 1.0f);
        }
    }

    // 과제 요구사항: 최소 20회 실행하여 평균 시간 측정
    int num_runs = 20;
    double total_execution_time_ms = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        // 매 실행마다 C 행렬 초기화
        std::fill(C.begin(), C.end(), 0.0f);

        auto run_start = std::chrono::high_resolution_clock::now();
        matrixMultiply_CPU(A, B, C, N, M, K);
        auto run_end = std::chrono::high_resolution_clock::now();
        
        total_execution_time_ms += std::chrono::duration<double, std::milli>(run_end - run_start).count();
    }

    double average_execution_time_ms = total_execution_time_ms / num_runs;

    // 정확성 검증을 위한 코너 요소 출력
    if (size <= 64) { // 작은 크기에서만 출력
        std::cout << "  검증: C[0][0]=" << C[0] << ", C[0][" << K-1 << "]=" << C[K-1] 
                  << ", C[" << N-1 << "][0]=" << C[(N-1)*K] << ", C[" << N-1 << "][" << K-1 << "]=" << C[(N-1)*K + (K-1)] << std::endl;
    }

    return average_execution_time_ms;
}

// 메인 함수
int main() {
    // 테스트할 행렬 크기 배열
    std::vector<int> matrix_sizes = {16, 32, 64, 128, 512, 1024};
    
    std::cout << "===== CPU 기반 행렬 곱셈 성능 측정 =====" << std::endl;
    std::cout << std::left << std::setw(12) << "행렬 크기" << std::setw(20) << "평균 실행 시간(ms)" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    // 각 크기에 대해 측정 실행
    for (int size : matrix_sizes) {
        std::cout << "행렬 크기 " << std::left << std::setw(4) << size << "x" << std::setw(4) << size << " 측정 중..." << std::endl;
        double avg_time = measureMatrixMultiplication(size);
        std::cout << std::left << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size)) 
                  << std::setw(20) << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }
    
    return 0;
}
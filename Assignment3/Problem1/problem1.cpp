#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (1024)
#define ALIGN (32)

void print_array(const int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128_u32(__m128 var, int start) {
  alignas(ALIGN) float val[4];
  _mm_store_ps(val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print256_u32(__m256 var, int start) {
  alignas(ALIGN) float val[8];
  _mm256_store_ps(val, var);
  cout << "Values [" << start <<":"<< start+7 << "]: " << val[0] << " " << val[1] << " "
       << val[2] << " " << val[3] << " " << val[4] << " "<< val[5] << " "
       << val[6] << " " << val[7] << "\n";
}

void print128_u64(__m128 var) {
  alignas(ALIGN) float val[2];
  _mm_store_ps(val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

void matmul_seq(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4(float** A, float** B, float** C) {

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      __m128 sum_vec = _mm_setzero_ps(); 
      for (int k = 0; k < N; k+=4){
        __m128 row_vec = _mm_loadu_ps(&A[i][k]); 
        __m128 col_vec = _mm_set_ps(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);
        // __m128 b0 = _mm_set1_ps(B[k][j]);
        // __m128 b1 = _mm_set1_ps(B[k+1][j]);
        // __m128 b2 = _mm_set1_ps(B[k+2][j]);
        // __m128 b3 = _mm_set1_ps(B[k+3][j]);
        __m128 multiplied_vec = _mm_mul_ps(row_vec, col_vec); 
        sum_vec = _mm_add_ps(sum_vec, multiplied_vec);
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b0));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b1));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b2));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b3));
      }
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec); // _mm_hadd_ps([a,b,c,d], [i,j,k,l]) = [a+b, c+d, i+j, k+l] -> This is how _mm_hadd_ps works.
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec); // doing _mm_hadd_ps twice on sum_vec makes all elements of sum_vec the sum of its original elements
      C[i][j] = _mm_cvtss_f32(sum_vec); 
      // _mm_store_ss(&C[i][j], sum_vec);
    }
  }
}

void matmul_sse4_aligned(float** A, float** B, float** C) {
  __builtin_assume_aligned(A, 16);
  __builtin_assume_aligned(B, 16);
  __builtin_assume_aligned(C, 16);
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      __m128 sum_vec = _mm_setzero_ps();
      for (int k = 0; k < N; k+=4){
        __m128 row_vec = _mm_load_ps(&A[i][k]);
        __m128 col_vec = _mm_set_ps(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]); 
        __m128 multiplied_vec = _mm_mul_ps(row_vec, col_vec); 
        // __m128 b0 = _mm_set1_ps(B[k][j]);
        // __m128 b1 = _mm_set1_ps(B[k+1][j]);
        // __m128 b2 = _mm_set1_ps(B[k+2][j]);
        // __m128 b3 = _mm_set1_ps(B[k+3][j]);
        sum_vec = _mm_add_ps(sum_vec, multiplied_vec);
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b0));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b1));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b2));
        // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b3));
      }
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec); 
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec); 
      C[i][j] = _mm_cvtss_f32(sum_vec); 
      // _mm_store_ss(&C[i][j], sum_vec);
    }
  }
}

void matmul_sse4_blocked(float** A, float** B, float** C) {
  __builtin_assume_aligned(A, 16);
  __builtin_assume_aligned(B, 16);
  __builtin_assume_aligned(C, 16);
  int block_size = 32;
  for (int i = 0; i < N; i += block_size) {
    for (int j = 0; j < N; j += block_size) {
      for (int k = 0; k < N; k += block_size){
        for (int ii = i; ii < i + block_size; ++ii) {
          for (int jj = j; jj < j + block_size; ++jj) {
            float sum = 0;
            __m128 sum_vec = _mm_setzero_ps();
            
            for (int kk = k; kk < k + block_size; kk+=4) {
                __m128 row_vec = _mm_load_ps(&A[ii][kk]);
                __m128 col_vec = _mm_set_ps(B[kk+3][jj], B[kk+2][jj], B[kk+1][jj], B[kk][jj]);
                __m128 multiplied_vec = _mm_mul_ps(row_vec, col_vec);
                // __m128 b0 = _mm_set1_ps(B[k][j]);
                // __m128 b1 = _mm_set1_ps(B[k+1][j]);
                // __m128 b2 = _mm_set1_ps(B[k+2][j]);
                // __m128 b3 = _mm_set1_ps(B[k+3][j]);
                sum_vec = _mm_add_ps(sum_vec, multiplied_vec);
                // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b0));
                // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b1));
                // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b2));
                // sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(row_vec, b3));
            }
            // C[ii][jj] += sum;
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            C[ii][jj] += _mm_cvtss_f32(sum_vec);
          }
        }
      }
    }
  }
}

void matmul_avx2(float** A, float** B, float** C) {
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      __m256 sum_vec = _mm256_setzero_ps(); 
      for (int k = 0; k < N; k+=8){
        __m256 row_vec = _mm256_loadu_ps(&A[i][k]);
        __m256 col_vec = _mm256_set_ps(B[k+7][j], B[k+6][j], B[k+5][j], B[k+4][j], B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);
        __m256 multiplied_vec = _mm256_mul_ps(row_vec, col_vec); 
        // __m256 b0 = _mm256_set1_ps(B[k][j]);
        // __m256 b1 = _mm256_set1_ps(B[k+1][j]);
        // __m256 b2 = _mm256_set1_ps(B[k+2][j]);
        // __m256 b3 = _mm256_set1_ps(B[k+3][j]);
        // __m256 b4 = _mm256_set1_ps(B[k+4][j]);
        // __m256 b5 = _mm256_set1_ps(B[k+5][j]);
        // __m256 b6 = _mm256_set1_ps(B[k+6][j]);
        // __m256 b7 = _mm256_set1_ps(B[k+7][j]);
        sum_vec = _mm256_add_ps(sum_vec, multiplied_vec);
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b0));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b1));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b2));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b3));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b4));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b5));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b6));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b7));
      }
      __m128 low = _mm256_castps256_ps128(sum_vec); 
      __m128 high = _mm256_extractf128_ps(sum_vec, 1); 
      __m128 sum128 = _mm_add_ps(high, low);
      sum128 = _mm_hadd_ps(sum128, sum128); 
      sum128 = _mm_hadd_ps(sum128, sum128); 
      C[i][j] = _mm_cvtss_f32(sum128); 
      // _mm_store_ss(&C[i][j], sum128);
    }
  }
}

void matmul_avx2_aligned(float** A, float** B, float** C) {
  __builtin_assume_aligned(A, 32);
  __builtin_assume_aligned(B, 32);
  __builtin_assume_aligned(C, 32);
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      __m256 sum_vec = _mm256_setzero_ps(); 
      for (int k = 0; k < N; k+=8){
        __m256 row_vec = _mm256_load_ps(&A[i][k]); 
        __m256 col_vec = _mm256_set_ps(B[k+7][j], B[k+6][j], B[k+5][j], B[k+4][j], B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);
        __m256 multiplied_vec = _mm256_mul_ps(row_vec, col_vec); 
        // __m256 b0 = _mm256_set1_ps(B[k][j]);
        // __m256 b1 = _mm256_set1_ps(B[k+1][j]);
        // __m256 b2 = _mm256_set1_ps(B[k+2][j]);
        // __m256 b3 = _mm256_set1_ps(B[k+3][j]);
        // __m256 b4 = _mm256_set1_ps(B[k+4][j]);
        // __m256 b5 = _mm256_set1_ps(B[k+5][j]);
        // __m256 b6 = _mm256_set1_ps(B[k+6][j]);
        // __m256 b7 = _mm256_set1_ps(B[k+7][j]);
        sum_vec = _mm256_add_ps(sum_vec, multiplied_vec);
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b0));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b1));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b2));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b3));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b4));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b5));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b6));
        // sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(row_vec, b7));
      }
      __m128 low = _mm256_castps256_ps128(sum_vec); 
      __m128 high = _mm256_extractf128_ps(sum_vec, 1); 
      __m128 sum128 = _mm_add_ps(high, low);
      sum128 = _mm_hadd_ps(sum128, sum128); 
      sum128 = _mm_hadd_ps(sum128, sum128); 
      C[i][j] = _mm_cvtss_f32(sum128); 
      // _mm_store_ss(&C[i][j], sum128);
    }
  }
}

void matmul_avx2_blocked(float** A, float** B, float** C) {
  __builtin_assume_aligned(A, 32);
  __builtin_assume_aligned(B, 32);
  __builtin_assume_aligned(C, 32);
  int block_size = 32;
  for (int i = 0; i < N; i += block_size) {
    for (int j = 0; j < N; j += block_size) {
      for (int k = 0; k < N; k += block_size){
        for (int ii = i; ii < i + block_size; ++ii) {
          for (int jj = j; jj < j + block_size; ++jj) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int kk = k; kk < k + block_size; kk+=4) {
                __m256 row_vec = _mm256_load_ps(&A[ii][kk]);
                __m256 col_vec = _mm256_set_ps(B[k+7][j], B[k+6][j], B[k+5][j], B[k+4][j], B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]); 
                __m256 multiplied_vec = _mm256_mul_ps(row_vec, col_vec);
                sum_vec = _mm256_add_ps(sum_vec, multiplied_vec);
            }
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(high, low);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            C[ii][jj] += _mm_cvtss_f32(sum128);
          }
        }
      }
    }
  }
}

void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  auto** A = new float*[N];
  float** A_aligned_16 = static_cast<float**>(std::aligned_alloc(16, sizeof(float*) * N));
  float** A_aligned_32 = static_cast<float**>(std::aligned_alloc(32, sizeof(float*) * N));
  for (int i = 0; i < N; i++) {
    A[i] = new float[N]();
    A_aligned_16[i] = static_cast<float*>(std::aligned_alloc(16, sizeof(float) * N));
    A_aligned_32[i] = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N));
  }


  auto** B = new float*[N];
  float** B_aligned_16 = static_cast<float**>(std::aligned_alloc(16, N * sizeof(float*)));
  float** B_aligned_32 = static_cast<float**>(std::aligned_alloc(32, N * sizeof(float*)));
  for (int i = 0; i < N; i++) {
    B[i] = new float[N]();
    B_aligned_16[i] = static_cast<float*>(std::aligned_alloc(16, sizeof(float) * N));
    B_aligned_32[i] = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N));
  }


  auto** C_seq = new float*[N];
  auto** C_sse4 = new float*[N];
  auto** C_avx2 = new float*[N];
  float** C_sse4_aligned = static_cast<float**>(std::aligned_alloc(16, sizeof(float*) * N));
  float** C_avx2_aligned = static_cast<float**>(std::aligned_alloc(32, sizeof(float*) * N));
  float** C_sse4_blocked = static_cast<float**>(std::aligned_alloc(32, sizeof(float*) * N));
  float** C_avx2_blocked = static_cast<float**>(std::aligned_alloc(32, sizeof(float*) * N));
  for (int i = 0; i < N; i++) {
    C_seq[i] = new float[N]();
    C_sse4[i] = new float[N]();
    C_avx2[i] = new float[N]();
    C_sse4_aligned[i] = static_cast<float*>(std::aligned_alloc(16, sizeof(float) * N));
    C_avx2_aligned[i] = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N));
    C_sse4_blocked[i] = static_cast<float*>(std::aligned_alloc(16, sizeof(float) * N));
    C_avx2_blocked[i] = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N));

  }

  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0.1F;
      A_aligned_16[i][j] = 0.1F;
      A_aligned_32[i][j] = 0.1F;
      B[i][j] = 0.2F;
      B_aligned_16[i][j] = 0.2F;
      B_aligned_32[i][j] = 0.2F;
      C_seq[i][j] = 0.0F;
      C_sse4[i][j] = 0.0F;
      C_sse4_aligned[i][j] = 0.0F;
      C_avx2[i][j] = 0.0F;
      C_avx2_aligned[i][j] = 0.0F;
    }
  }

  HRTimer start = HR::now();
  matmul_seq(A, B, C_seq);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4(A, B, C_sse4);
  end = HR::now();
  check_result(C_seq, C_sse4);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4_aligned(A_aligned_16, B_aligned_16, C_sse4_aligned);
  end = HR::now();
  check_result(C_seq, C_sse4_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4_aligned time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4_blocked(A_aligned_16, B_aligned_16, C_sse4_blocked);
  end = HR::now();
  check_result(C_seq, C_sse4_blocked);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4_blocked time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2(A, B, C_avx2);
  end = HR::now();
  check_result(C_seq, C_avx2);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_aligned(A_aligned_32, B_aligned_32, C_avx2_aligned);
  end = HR::now();
  check_result(C_seq, C_avx2_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2_aligned time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_blocked(A_aligned_32, B_aligned_32, C_avx2_blocked);
  end = HR::now();
  check_result(C_seq, C_avx2_blocked);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2_blocked time: " << duration << " ms" << endl;

  return EXIT_SUCCESS;
}

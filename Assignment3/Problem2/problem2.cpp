#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 30)
#define SSE_WIDTH_BITS (128)
#define AVX_WIDTH_BITS (256)
#define ALIGN (32)

/** Helper methods for debugging */

void print_array(const int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print256i_u32(__m256i var, int start) {
  alignas(ALIGN) uint32_t val[8];
  _mm256_store_si256((__m256i*)val, var);
  cout << "Values [" << start <<":"<< start+7 << "]: " << val[0] << " " << val[1] << " "
       << val[2] << " " << val[3] << " " << val[4] << " "<< val[5] << " "
       << val[6] << " " << val[7] << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

__attribute__((optimize("no-tree-vectorize"))) int ref_version(int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp)
  for (int i = 0; i < N; i++) {
    tmp += source[i];
#pragma omp scan inclusive(tmp)
    dest[i] = tmp;
  }
  return tmp;
}

// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds,
// one broadcast
int sse4_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);
  
  // Return vector of type __m128i with all elements set to zero, to be added as
  // previous sum for the first four elements.
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be
    // aligned on a 16-byte boundary to be safe.
    __m128i x = _mm_load_si128((__m128i*)&source[i]);
    // Let the numbers in x be [d,c,b,a], where a is at source[i].
    __m128i tmp0 = _mm_slli_si128(x, 4);
    // Shift x left by 4 bytes while shifting in zeros. tmp0 becomes [c,b,a,0].
    __m128i tmp1 =
        _mm_add_epi32(x, tmp0); // Add packed 32-bit integers in x and tmp0.
    // tmp1 becomes [c+d,b+c,a+b,a].
    // Shift tmp1 left by 8 bytes while shifting in zeros.
    __m128i tmp2 = _mm_slli_si128(tmp1, 8); // tmp2 becomes [a+b,a,0,0].
    // Add packed 32-bit integers in tmp2 and tmp1.
    __m128i out = _mm_add_epi32(tmp2, tmp1);
    // out contains [a+b+c+d,a+b+c,a+b,a].
    out = _mm_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by
    // offset.
    // Store 128-bits of integer data from out into memory. dest_addr must be
    // aligned on a 16-byte boundary to be safe.
    _mm_store_si128((__m128i*)&dest[i], out);
    // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the
    // formula (z << 6) | (y << 4) | (x << 2) | w.
    // int mask = _MM_SHUFFLE(3, 3, 3, 3);
    // Bits [7:0] of mask are 11111111 to pick the third integer (11) from out
    // (i.e., a+b+c+d).

    // Shuffle 32-bit integers in out using the control in mask.
    offset = _mm_shuffle_epi32(out, _MM_SHUFFLE(3, 3, 3, 3));
    // offset now contains 4 copies of a+b+c+d.
  }
  return dest[N - 1];
}

int avx2_version(const int* __restrict__ source, int* __restrict__ dest) { 
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  // offset stores prefix sum upto previous vector calculation
  // initialized with 8 zeros
  __m256i offset = _mm256_setzero_si256();
  const int stride = AVX_WIDTH_BITS/(sizeof(int)*CHAR_BIT);

  // 
  for(int i=0; i<N; i+=stride){
    __m256i x = _mm256_load_si256((__m256i*)&source[i]); // let x = [a,b,c,d,e,f,g,h]
    // x = _mm256_set_epi32(8,7,6,5,4,3,2,1);
    // print256i_u32(x,0);
    __m256i tmp0 = _mm256_slli_si256(x, 4);
    __m256i tmp1 = _mm256_add_epi32(x, tmp0);
    __m256i tmp2 = _mm256_slli_si256(tmp1, 8);
    __m256i tmp3 = _mm256_add_epi32(tmp1, tmp2); // now tmp3 = [a,a+b,a+b+c,a+b+c+d,e,e+f,e+f+g,e+f+g+h]
    // print256i_u32(tmp3,0);
    __m256i left_half_sum = _mm256_set1_epi32 (_mm256_extract_epi32 (tmp3, 3)); // extracts a+b+c+d
    // print256i_u32(left_half_sum,0);
    __m256i zeros = _mm256_setzero_si256();
    __m256i mask = _mm256_blend_epi32 (left_half_sum, zeros, 0x0F); // creates a mask of [0,0,0,0,a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d]
    __m256i out = _mm256_add_epi32(tmp3,mask); // adding tmp3 and mask creates full prefix sum of vector
    // print256i_u32(out,0);
    out = _mm256_add_epi32(out, offset);
    _mm256_store_si256((__m256i*)&dest[i],out);
    mask = _mm256_set1_epi32(7);
    offset = _mm256_permutevar8x32_epi32(out, mask); 
    // cout<<"-----------------"<<endl;
  }
  return dest[N-1]; 
}

__attribute__((optimize("no-tree-vectorize"))) int main() {
  int* array = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(array, array + N, 1);

  int* ref_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);
  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  int* omp_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(omp_res, omp_res + N, 0);
  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;
  free(omp_res);

  int* sse_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(sse_res, sse_res + N, 0);
  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
  cout << "SSE version: " << val_sse << " time: " << duration << endl;
  free(sse_res);

  int* avx2_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  int val_avx2 = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx2 << " time: " << duration << endl;

  return EXIT_SUCCESS;
}

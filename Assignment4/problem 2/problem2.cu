#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h> 
#include <thrust/scan.h>

#define THRESHOLD (std::numeric_limits<double>::epsilon())
#define int32 uint32_t

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (1 << 10);
const uint64_t TPB = 256;

__host__ void thrust_sum(const uint32_t* input, uint32_t* output) {
  thrust::exclusive_scan(input, input+N, output);
}

__global__ void cuda_sum(int32 *arr_in, int32 *out) {
  extern __shared__ int32 temp[];
  int32 thid = threadIdx.x;
  int32 bid = blockIdx.x;
  int32 th_num = blockDim.x;
  int32 tid = bid * th_num + thid;

  int offset = 1;
  if (tid < N) temp[thid] = arr_in[tid];
  else temp[thid] = 0;

  // up-sweep phase
  for(int d = th_num>>1; d>0; d>>=1){
    __syncthreads();
    if(thid < d){
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *=2;
  }

  // Down-sweep phase
  if(thid == 0) temp[th_num-1] = 0;
  for(int d=1; d<th_num; d*=2){
    offset >>=1;
    __syncthreads();
    if(thid<d){
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  out[tid] = temp[thid];
  __syncthreads();
  int32 block_sum = out[th_num-1]+1;
  out[tid] += (bid)*block_sum;
}



// #define CONFLICT_FREE_OFFSET(n) \
//   ((n) >> 7)

// __global__ void cuda_sum(int32 *arr_in, int32 *out) {

//   extern __shared__ int32 temp[];
//   int32 thid = threadIdx.x;
//   int32 bid = blockIdx.x;
//   int32 th_num = blockDim.x;
//   int32 tid = bid * th_num + thid;

//   int offset = 1;
//   if (tid < N){
//     int ai = thid;
//     int bi = thid + (th_num/2);

//     int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
//     int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

//     temp[ai + bankOffsetA] = arr_in[bid*th_num+ai];
//     temp[bi + bankOffsetB] = arr_in[bid*th_num+bi];
//   }
//   else temp[thid] = 0;

//   // up-sweep phase
//   for(int d = th_num>>1; d>0; d>>=1){
//     __syncthreads();
//     if(thid < d){
//       int ai = offset*(2*thid+1)-1;
//       int bi = offset*(2*thid+2)-1;
//       ai += CONFLICT_FREE_OFFSET(ai);
//       bi += CONFLICT_FREE_OFFSET(bi);
//       temp[bi] += temp[ai];
//     }
//     offset *=2;
//   }

//   // Down-sweep phase
//   if(thid == 0) temp[th_num-1 + CONFLICT_FREE_OFFSET(th_num-1)] = 0;
//   for(int d=1; d<th_num; d*=2){
//     offset >>=1;
//     __syncthreads();
//     if(thid<d){
//       int ai = offset*(2*thid+1)-1;
//       int bi = offset*(2*thid+2)-1;
//       ai += CONFLICT_FREE_OFFSET(ai);
//       bi += CONFLICT_FREE_OFFSET(bi);
//       float t = temp[ai];
//       temp[ai] = temp[bi];
//       temp[bi] += t;
//     }
//   }
//   __syncthreads();
//   out[bid*th_num+thid] = temp[thid + CONFLICT_FREE_OFFSET(thid)];
//   out[bid*th_num+(thid + (th_num/2))] = temp[(thid + (th_num/2)) + CONFLICT_FREE_OFFSET(thid + (th_num/2))];
//   __syncthreads();
//   int32 block_sum = out[th_num-1]+1;
//   out[tid] += (bid)*block_sum;
// }

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                           const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between the two arrays.\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}


void print_arr(int32* arr, int32 n){
  cout<<arr[0];
  for(int32 i=1;i<n;i++){
    cout<<", "<<arr[i];
  }
  cout<<endl;
}

int main() {
  auto* h_input = new uint32_t[N];
  std::fill_n(h_input, N, 1);

  // Use Thrust code as reference
  auto* h_thrust_ref = new uint32_t[N];
  std::fill_n(h_thrust_ref, N, 0);

  // TODO: Time your code
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  thrust_sum(h_input, h_thrust_ref);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float t_time = 0.0f;
  cudaEventElapsedTime(&t_time, start, end);

  // print_arr(h_thrust_ref, N);

  // TODO: Use a CUDA kernel, time your code
  int32 *arr_in = (int32*)malloc(sizeof(int32)*N); 
  int32 *arr_out = (int32*)malloc(sizeof(int32)*N);
  std::fill_n(arr_in, N, 1);
  std::fill_n(arr_out, N, 0);

  int32 *d_arr_in, *d_arr_out;
  cudaCheckError(cudaMalloc((void**)&d_arr_in, sizeof(int32)*N));
  cudaCheckError(cudaMalloc((void**)&d_arr_out, sizeof(int32)*N));
  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(d_arr_in, arr_in, sizeof(int32)*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float h2d = 0.0f;
  cudaEventElapsedTime(&h2d, start, end);  

  cudaEventRecord(start, 0);
  cuda_sum<<<(N+TPB-1)/TPB, TPB>>>(d_arr_in, d_arr_out);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(arr_out, d_arr_out, sizeof(int32)*N, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float d2h = 0.0f;
  cudaEventElapsedTime(&d2h, start, end);

  // print_arr(arr_out, N);

  check_result(h_thrust_ref, arr_out, N);
  cout<<"Time taken by Thrust version : "<<t_time<<endl<<endl;
  cout<<"Time taken by host to device memcpy : "<<h2d<<endl;
  cout<<"Time taken by cuda Kernel : "<<kernel_time<<endl;
  cout<<"Time taken by device to host memcpy : "<<d2h<<endl<<endl;

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(d_arr_in);
  cudaFree(d_arr_out);
  free(arr_in);
  free(arr_out);
  delete[] h_thrust_ref;
  delete[] h_input;

  return EXIT_SUCCESS;
}

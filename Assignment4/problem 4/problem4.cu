#include <cstdlib>
#include <cuda.h>
#include <iostream> 
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

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

const uint64_t N = (1 << 6);
const uint64_t chunk = 8;

// TODO: Edit the function definition as required
__global__ void kernel2D(float *input, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= N || y >= N) return;
  
  float sum = 0.0f;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int nx = x + j;
      int ny = y + i; 
      if(!(i||j)) continue; // only have to calculate avg of neighboring cells not current cell
      if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
        sum += input[ny * N + nx];
      }
    }
  }
  output[y * N + x] = sum / 8.0f;
}



__global__ void kernel2D_opt(float *input, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t tid = y * N + x;

  if(x >= N || y >= N) return;

  int x_ = threadIdx.x+1, y_ = threadIdx.y+1; // x_ and y_ will point to required memory locations in shared memory tile

  __shared__ float temp[chunk+2][chunk+2];
  temp[x_][y_] = input[tid];

  // filling the corner ghost cells of the temp
  if(threadIdx.x == 0 && threadIdx.y == 0){
    temp[0][0] = (x == 0 || y == 0) ? 0.0f : input[(y - 1) * N + (x - 1)];
    temp[0][chunk + 1] = (x == 0 || (y + chunk) >= N) ? 0.0f : input[(y + chunk) * N + (x - 1)];
    temp[chunk + 1][0] = ((x + chunk) >= N || y == 0) ? 0.0f : input[(y - 1) * N + (x + chunk)];
    temp[chunk + 1][chunk + 1] = ((x + chunk) >= N || (y + chunk) >= N) ? 0.0f : input[(y + chunk) * N + (x + chunk)];       
  }
  // filling ghost cells on the edges
  if(threadIdx.x == 0){
    temp[0][y_] = (x == 0) ? 0.0f : input[(y * N) + (x - 1)];
    temp[chunk + 1][y_] = ((x + chunk) >= N) ? 0.0f : input[(y * N) + (x + chunk)];
  }
  if(threadIdx.y == 0){
    temp[x_][0] = (y == 0) ? 0.0f : input[(y - 1) * N + x];
    temp[x_][chunk + 1] = ((y + chunk) >= N) ? 0.0f : input[(y + chunk) * N + x];
  }
  __syncthreads();
  
  float sum = temp[x_ - 1][y_ - 1] + temp[x_ - 1][y_] + temp[x_ - 1][y_ + 1] +
              temp[x_][y_ - 1] + temp[x_][y_ + 1] +
              temp[x_ + 1][y_ - 1] + temp[x_ + 1][y_] + temp[x_ + 1][y_ + 1];
  output[tid] = sum / 8.0f;
}

// TODO: Edit the function definition as required
__global__ void kernel3D(float *input, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= N || y >= N || z >= N) return; 
  float sum = 0.0;
  
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        int nx = x + i;
        int ny = y + j;
        int nz = z + k;
        if (i == 0 && j == 0 && k == 0) continue; // Skip the center cell itself
        if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N) {
          sum += input[nz * N * N + ny * N + nx];
        }
      }
    }
  }

  output[z * N * N + y * N + x] = sum / 26.0f;
}



__global__ void kernel3D_opt(float *input, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int tid = z*N*N + y * N + x;
  if(x >= N || y >= N || z >= N) return;

  int x_ = threadIdx.x+1, y_ = threadIdx.y+1, z_ = threadIdx.z+1; // x_, y_ and z_ will point to required memory locations in shared memory tile

  __shared__ float temp[chunk+2][chunk+2][chunk+2];
  temp[x_][y_][z_] = input[tid];

  // loading ghost cells for the cube face
  if(threadIdx.x == 0){
    temp[0][y_][z_] = (x == 0)? 0 : input[tid-1];
    temp[chunk + 1][y_][z_] = ((x+chunk) >= N)? 0 : input[tid+chunk];
  }
  if(threadIdx.y == 0){
    temp[x_][0][z_] = (y == 0)? 0 : input[tid-N];
    temp[x_][chunk + 1][z_] = ((y+chunk) >= N)? 0 : input[tid+chunk*N];
  }
  if(threadIdx.z == 0){
    temp[x_][y_][0] = (z == 0)? 0 : input[tid-N*N];
    temp[x_][y_][chunk + 1] = ((z+chunk) >= N)? 0 : input[tid+chunk*N*N];
  }

  // loading ghost cells for the edges
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    temp[0][0][z_] = (x == 0 || y == 0) ? 0.0f : input[tid - N - 1];
    temp[chunk + 1][0][z_] = ((x + chunk) >= N || y == 0) ? 0.0f : input[tid - N + chunk];
    temp[0][chunk + 1][z_] = (x == 0 || (y + chunk) >= N) ? 0.0f : input[tid + N*chunk - 1];
    temp[chunk + 1][chunk + 1][z_] = ((x + chunk) >= N || (y + chunk) >= N) ? 0.0f : input[tid + N*chunk + chunk];
  }
  if (threadIdx.y == 0 && threadIdx.z == 0) {
    temp[x_][0][0] = (y == 0 || z == 0) ? 0.0f : input[tid - N*N - N];
    temp[x_][chunk + 1][0] = ((y + chunk) >= N || z == 0) ? 0.0f : input[tid - N*N + N*chunk];
    temp[x_][0][chunk + 1] = (y == 0 || (z + chunk) >= N) ? 0.0f : input[tid + chunk*N*N - N];
    temp[x_][chunk + 1][chunk + 1] = ((y + chunk) >= N || (z + chunk) >= N) ? 0.0f : input[tid + chunk*N*N + N*chunk];
  }
  if (threadIdx.x == 0 && threadIdx.z == 0) {
    temp[0][y_][0] = (x == 0 || z == 0) ? 0.0f : input[tid - N*N - 1];
    temp[chunk + 1][y_][0] = ((x + chunk) >= N || z == 0) ? 0.0f : input[tid - N*N + chunk];
    temp[0][y_][chunk + 1] = (x == 0 || (z + chunk) >= N) ? 0.0f : input[tid + chunk*N*N - 1];
    temp[chunk + 1][y_][chunk + 1] = ((x + chunk) >= N || (z + chunk) >= N) ? 0.0f : input[tid + chunk*N*N + chunk];
  }

  // loading ghost cells for the corners
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    temp[0][0][0] =                         (x==0 || y==0 || z==0) ? 0.0f : input[(z-1)*N*N + (y-1)*N + (x-1)];
    temp[0][0][chunk + 1] =                 (x==0 || y==0 || (z + chunk)>=N) ? 0.0f : input[(z+chunk)*N*N + (y-1)*N + (x-1)];
    temp[0][chunk + 1][0] =                 (x==0 || (y + chunk)>=N || z==0) ? 0.0f : input[(z-1)*N*N + (y + chunk)*N + (x-1)];
    temp[0][chunk + 1][chunk + 1] =         (x==0 || (y + chunk)>=N || (z + chunk)>=N) ? 0.0f : input[(z + chunk)*N*N + (y + chunk)*N + (x-1)];
    temp[chunk + 1][0][0] =                 ((x + chunk)>=N || y==0 || z==0) ? 0.0f : input[(z-1)*N*N + (y-1) * N + (x + chunk)];
    temp[chunk + 1][0][chunk + 1] =         ((x + chunk)>=N || y==0 || (z + chunk)>=N) ? 0.0f : input[(z + chunk)*N*N + (y-1)*N + (x + chunk)];
    temp[chunk + 1][chunk + 1][0] =         ((x + chunk)>=N || (y + chunk)>=N || z==0) ? 0.0f : input[(z-1)*N*N + (y + chunk)*N + (x + chunk)];
    temp[chunk + 1][chunk + 1][chunk + 1] = ((x + chunk)>=N || (y + chunk)>=N || (z + chunk)>=N) ? 0.0f : input[(z + chunk)*N*N + (y + chunk)*N + (x + chunk)];
  }
__syncthreads();

  float sum = 0.0f;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        if (i || j || k) { // Exclude the center cell
            sum += temp[x_ + i][y_ + j][z_ + k];
        }
      }
    }
  }

  output[z * N * N + y * N + x] = sum / 26.0f;
}


__host__ void convolution2D(float* input, float* output) {
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      double sum = 0.0;

      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          if (i == 0 && j == 0) continue;  // Skip the center cell

          int nx = x + i;
          int ny = y + j;
          
          if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
            sum += input[ny*N + nx];
          }
        }
      }
      output[y*N + x] = sum / 8.0f;
    }
  }
}

__host__ void convolution3D(float* input, float* output) {
  for (int z = 0; z < N; z++) {
    for (int y = 0; y < N; y++) {
      for (int x = 0; x < N; x++) {
        float sum = 0.0f;

        for (int k = -1; k <= 1; k++) {
          for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
              if (i == 0 && j == 0 && k == 0) continue; // skip the center cell

              int nx = x + i;
              int ny = y + j;
              int nz = z + k;

              // Check bounds and accumulate neighbors
              if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N) {
                sum += input[(nz * N * N) + (ny * N) + nx];
              }
            }
          }
        }

        output[(z * N * N) + (y * N) + x] = sum / 26.0f;
      }
    }
  }
}

__host__ void check_result_2d(const float* w_ref, const float* w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      float this_diff =
          w_ref[i + N * j] - w_opt[i + N * j];
      if (std::fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff) {
          maxdiff = this_diff;
        }
      }      
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ void check_result_3d(const float* w_ref, const float* w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        float this_diff =
            w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print2D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "n";
  }
}

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i*N*N + j * N + k] << "\t";
      }
      cout << "n";
    }
    cout << "n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  // uint64_t TPB = 64;
  srand(time(NULL));

  cout<<"tile lenght for kernel using shared memory : "<<chunk<<endl;

  float* grid_in_2d = (float*)malloc(N*N*sizeof(float));
  float* grid_ref_2d = (float*)malloc(N*N*sizeof(float));
  float* grid_out_2d = (float*)malloc(N*N*sizeof(float));

  float* grid_in_3d = (float*)malloc(N*N*N*sizeof(float));
  float* grid_ref_3d = (float*)malloc(N*N*N*sizeof(float));
  float* grid_out_3d = (float*)malloc(N*N*N*sizeof(float));

  float *d_grid_in_2d, *d_grid_out_2d;
  float *d_grid_in_3d, *d_grid_out_3d;

  for(int i=0;i<N*N;i++){
    grid_in_2d[i] = (float)((rand()%10)+1)/10;
    grid_ref_2d[i] = 0.0f;
    grid_out_2d[i] = 0.0f;
  }
  for(int i=0;i<N*N*N;i++){
    grid_in_3d[i] = (float)((rand()%10)+1)/10;
    grid_ref_3d[i] = 0.0f;
    grid_out_3d[i] = 0.0f;
  }

  cudaCheckError(cudaMalloc((void**)&d_grid_in_2d, sizeof(float)*N*N));
  cudaCheckError(cudaMalloc((void**)&d_grid_out_2d, sizeof(float)*N*N));
  cudaCheckError(cudaMalloc((void**)&d_grid_in_3d, sizeof(float)*N*N*N));
  cudaCheckError(cudaMalloc((void**)&d_grid_out_3d, sizeof(float)*N*N*N));

  double clkbegin = rtclock();
  convolution2D(grid_in_2d,grid_ref_2d);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "2d kernel time on CPU: " << cpu_time * 1000 << " msec" << endl << endl;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // TODO: Fill in kernel2D
  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(d_grid_in_2d, grid_in_2d, sizeof(float)*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float h2d = 0.0f;
  cudaEventElapsedTime(&h2d, start, end);

  dim3 blocksize(chunk, chunk, 1);
  dim3 gridsize((N+chunk-1)/chunk, (N+chunk-1)/chunk, 1);
  cudaEventRecord(start, 0);
  kernel2D<<<gridsize,blocksize>>>(d_grid_in_2d, d_grid_out_2d);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_out_2d, d_grid_out_2d, sizeof(float)*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float d2h = 0.0f;
  cudaEventElapsedTime(&d2h, start, end);

  // TODO: Adapt check_result() and invoke
  check_result_2d(grid_ref_2d, grid_out_2d);
  std::cout << "host to device memcpy time (ms) : " << h2d<<endl;
  std::cout << "Kernel2D time (ms): " << kernel_time << "\n";
  std::cout << "device to host memcpy time (ms) : " <<d2h<<endl<<endl;

  // TODO: Fill in kernel2d_opt

  cudaEventRecord(start, 0);
  kernel2D_opt<<<gridsize,blocksize>>>(d_grid_in_2d, d_grid_out_2d);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_out_2d, d_grid_out_2d, sizeof(float)*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  d2h = 0.0f;
  cudaEventElapsedTime(&d2h, start, end);

  // TODO: Adapt check_result() and invoke
  check_result_2d(grid_ref_2d, grid_out_2d);

  std::cout << "host to device memcpy time (ms) : " << h2d<<endl;
  std::cout << "Optimized Kernel2D time (ms): " << kernel_time << "\n";
  std::cout << "device to host memcpy time (ms) : " <<d2h<<endl<<endl;

  // TODO: Fill in kernel3D

  clkbegin = rtclock();
  convolution3D(grid_in_3d,grid_ref_3d);
  clkend = rtclock();
  cpu_time = clkend - clkbegin;
  cout << "3d kernel time on CPU: " << cpu_time * 1000 << " msec" << endl << endl;

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(d_grid_in_3d, grid_in_3d, sizeof(float)*N*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  h2d = 0.0f;
  cudaEventElapsedTime(&h2d, start, end);

  dim3 blocksize2(chunk, chunk, chunk);
  dim3 gridsize2((N+chunk-1)/chunk, (N+chunk-1)/chunk, (N+chunk-1)/chunk);
  cudaEventRecord(start, 0);
  kernel3D<<<gridsize2,blocksize2>>>(d_grid_in_3d, d_grid_out_3d);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_out_3d, d_grid_out_3d, sizeof(float)*N*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  d2h = 0.0f;
  cudaEventElapsedTime(&d2h, start, end);

  // TODO: Adapt check_result() and invoke
  check_result_3d(grid_ref_3d, grid_out_3d);
  std::cout << "host to device memcpy time (ms) : " << h2d<<endl;
  std::cout << "Kernel3D time (ms): " << kernel_time << "\n";
  std::cout << "device to host memcpy time (ms) : " <<d2h<<endl<<endl;


  // TODO: Fill 3d optimized kernel
  cudaEventRecord(start, 0);
  kernel3D_opt<<<gridsize2,blocksize2>>>(d_grid_in_3d, d_grid_out_3d);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_out_3d, d_grid_out_3d, sizeof(float)*N*N*N, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  d2h = 0.0f;
  cudaEventElapsedTime(&d2h, start, end);

  // TODO: Adapt check_result() and invoke
  check_result_3d(grid_ref_3d, grid_out_3d);
  std::cout << "host to device memcpy time (ms) : " << h2d<<endl;
  std::cout << "Optimized Kernel3D time (ms): " << kernel_time << "\n";
  std::cout << "device to host memcpy time (ms) : " <<d2h<<endl<<endl;
  // TODO: Free memory

  return EXIT_SUCCESS;
}

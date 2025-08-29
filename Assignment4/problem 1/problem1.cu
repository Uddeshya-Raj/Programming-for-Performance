#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (64);
const uint64_t chunk = 8; // length of shared memory tile; shared memory allocated will be sizeof(double)*{(chunk+2)^3}

__device__ void print_mat_d(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%0.2lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

// TODO: Edit the function definition as required
__global__ void kernel1(double *grid_in, double *grid_out, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // if(tid == 0) print_mat_d(grid_out);
  uint64_t i = tid/(n*n);
  uint64_t j = (tid%(n*n))/n;
  uint64_t k = tid%n;
  if(tid >= n*n*n)return;
  if(i>0 && i<n-1 && j>0 && j<n-1 && k>0 && k<n-1) grid_out[i*n*n + j*n + k] = 0.8 * (grid_in[(i-1)*n*n +j*n + k]+grid_in[(i+1)*n*n + j*n + k]+grid_in[i*n*n + (j-1)*n + k]+grid_in[i*n*n + (j+1)*n + k]+grid_in[i*n*n + j*n + k-1]+grid_in[i*n*n + j*n + k+1]);
  else grid_out[i*n*n + j*n + k] = 0; // this is how I handle boundaries here
}

// TODO: Edit the function definition as required
__global__ void kernel2(double *grid_in, double *grid_out, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= n || y >= n || z >= n) return;

  uint64_t tid = x*n*n + y*n + z; 
  
  //coordinates to access shared memory (has +1 becauese of offset caused by ghost cells)
  int local_x = threadIdx.x+1;
  int local_y = threadIdx.y+1;
  int local_z = threadIdx.z+1; 

  // Shared memory tile with extra space for ghost cells
  __shared__ float s_grid[chunk+2][chunk+2][chunk+2];
  s_grid[local_x][local_y][local_z] = grid_in[tid];
  
  if(threadIdx.x == 0) s_grid[0][local_y][local_z] = grid_in[(x-1)*n*n + y*n + z]; // filling starting ghost cells in x-direction
  if(threadIdx.x == blockDim.x-1) s_grid[blockDim.x+1][local_y][local_z] = grid_in[(x+1)*n*n + y*n + z]; // filling last ghost cells in x-direction
  if(threadIdx.y == 0) s_grid[local_x][0][local_z] = grid_in[x*n*n + (y-1)*n + z]; 
  if(threadIdx.y == blockDim.y-1) s_grid[local_x][blockDim.y+1][local_z] = grid_in[x*n*n + (y+1)*n + z];
  if(threadIdx.z == 0) s_grid[local_x][local_y][0] = grid_in[x*n*n + y*n + z-1];
  if(threadIdx.z == blockDim.z-1) s_grid[local_x][local_y][blockDim.z+1] = grid_in[x*n*n + y*n + z+1];
  __syncthreads();

  if(!(x>0 && x<n-1 && y>0 && y<n-1 && z>0 && z<n-1)) grid_out[tid] = 0; //for boundaries of 3d grid
  else grid_out[tid] = 0.8 * (s_grid[local_x-1][local_y][local_z] + s_grid[local_x+1][local_y][local_z] 
                                                  + s_grid[local_x][local_y-1][local_z] + s_grid[local_x][local_y+1][local_z]
                                                  + s_grid[local_x][local_y][local_z-1] + s_grid[local_x][local_y][local_z+1]);
}

// TODO: Edit the function definition as required
__host__ void stencil(double *grid_in, double *grid_out, uint64_t n) {
  for(int i=1; i<n-1; i++){
    for(int j=1; j<n-1; j++){
      for(int k=1; k<n-1; k++){
        grid_out[i*n*n + j*n + k] = 0.8 * (grid_in[(i-1)*n*n +j*n + k]+grid_in[(i+1)*n*n + j*n + k]+grid_in[i*n*n + (j-1)*n + k]+grid_in[i*n*n + (j+1)*n + k]+grid_in[i*n*n + j*n + k-1]+grid_in[i*n*n + j*n + k+1]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            w_ref[i*N*N + j*N + k] - w_opt[i*N*N + j*N + k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (std::fabs(this_diff) > maxdiff) {
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

void print_mat(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%0.2lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
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
  uint64_t SIZE = N * N * N;
  uint64_t TPB = 64;
  srand(time(NULL));

  cout<<"tile lenght for kernel using shared memory : "<<chunk<<endl;

  double* grid_in = (double*)malloc(SIZE*sizeof(double));
  double* grid_ref = (double*)malloc(SIZE*sizeof(double));
  double* grid_out = (double*)malloc(SIZE*sizeof(double));
  double* grid_opt1 = (double*)malloc(SIZE*sizeof(double));
  double* grid_opt2 = (double*)malloc(SIZE*sizeof(double));


  double *d_grid_in, *d_grid_out;

  for(int i=0;i<SIZE;i++){
    grid_in[i] = (double)((rand()%10)+1)/10;
    grid_ref[i] = 0.0f;
    grid_opt1[i] = 0.0f;
  }

  // print_mat(grid_in);
  // cout<<endl<<endl;

  double clkbegin = rtclock();
  stencil(grid_in,grid_ref,N);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl << endl;

  // cout<<"ref:"<<endl;
  // print_mat(grid_ref);
  // cout<<endl<<endl;

  // cudaError_t status;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // TODO: Fill in kernel1 ======================================================================================================================================
  cudaCheckError(cudaMalloc((void**)&d_grid_in, sizeof(double)*SIZE));
  cudaCheckError(cudaMalloc((void**)&d_grid_out, sizeof(double)*SIZE));

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(d_grid_in, grid_in, sizeof(double)*SIZE, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float h2d_copy = 0.0f;
  cudaEventElapsedTime(&h2d_copy, start, end);

  cudaEventRecord(start, 0);
  kernel1<<<(SIZE+TPB-1/TPB), TPB>>>(d_grid_in, d_grid_out, N); // default kernel without optimizations
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_out, d_grid_out, sizeof(double)*SIZE, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float d2h_copy = 0.0f;
  cudaEventElapsedTime(&d2h_copy, start, end);


  // TODO: Adapt check_result() and invoke
  check_result(grid_ref, grid_out, N);
  std::cout << "Host to device Memcpy time (ms): "<< h2d_copy << endl;
  std::cout << "Default Kernel time (ms): " << kernel_time << endl;
  std::cout << "Device to host Memcpy time (ms): "<< d2h_copy <<endl<<endl;
  cudaFree(d_grid_out);
  // =============================================================================================================================================================


 
  // TODO: Fill in kernel2 =======================================================================================================================================
  cudaCheckError(cudaMalloc((void**)&d_grid_out, sizeof(double)*SIZE));

  uint64_t s_mem_size = ((chunk+2)*(chunk+2)*(chunk+2))*sizeof(double); // shared memory size
  dim3 blocksize(chunk, chunk, chunk);
  dim3 gridsize((N+chunk-1)/chunk, (N+chunk-1)/chunk, (N+chunk-1)/chunk);

  cudaEventRecord(start, 0);
  kernel2<<<gridsize, blocksize, s_mem_size>>>(d_grid_in, d_grid_out, N); // kernel with shared memory tiling
  // kernel1<<<(SIZE+TPB-1/TPB), TPB>>>(d_grid_in, d_grid_out, N); 
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_opt1, d_grid_out, sizeof(double)*SIZE, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  d2h_copy = 0.0f;
  cudaEventElapsedTime(&d2h_copy, start, end);

  // TODO: Adapt check_result() and invoke
  check_result(grid_ref,grid_opt1, N);

  std::cout << "Host to device Memcpy time (ms): "<< h2d_copy << endl;
  std::cout << "Memory tiled Kernel time (ms): " << kernel_time << endl;
  std::cout << "Device to host Memcpy time (ms): "<< d2h_copy <<endl<<endl;
  cudaFree(d_grid_out);
  // =============================================================================================================================================================
  


  // TODO: Fill in kernel3 =======================================================================================================================================
  
  // KERNEL 2 does not have a loop so no loop transfromation is required
  
  // TODO: Adapt check_result() and invoke
  // cudaEventElapsedTime(&kernel_time, start, end);
  // std::cout << "Kernel 3 time (ms): " << kernel_time << "\n";
  // =============================================================================================================================================================



  // TODO: Fill in kernel4 =======================================================================================================================================
  double *grid_opt3, *h_grid_in;
  cudaCheckError(cudaHostAlloc((void**)&grid_opt3, sizeof(double)*SIZE, cudaHostAllocDefault));
  cudaCheckError(cudaHostAlloc((void**)&h_grid_in, sizeof(double)*SIZE, cudaHostAllocDefault));
  for(int i=0;i<SIZE;i++) h_grid_in[i] = grid_in[i]; // so that same input is provided to both cpu and kernel versions

  cudaCheckError(cudaMalloc((void**)&d_grid_out, sizeof(double)*SIZE));

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(d_grid_in, h_grid_in, sizeof(double)*SIZE, cudaMemcpyHostToDevice));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  h2d_copy = 0.0f;
  cudaEventElapsedTime(&h2d_copy, start, end);

  // s_mem_size = ((chunk+2)*(chunk+2)*(chunk+2))*sizeof(double); // shared memory size
  // dim3 blocksize(chunk, chunk, chunk);
  // dim3 gridsize(N/chunk, N/chunk, N/chunk);

  cudaEventRecord(start, 0);
  kernel2<<<gridsize, blocksize, s_mem_size>>>(d_grid_in, d_grid_out, N);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  cudaEventRecord(start, 0);
  cudaCheckError(cudaMemcpy(grid_opt3, d_grid_out, sizeof(double)*SIZE, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  d2h_copy = 0.0f;
  cudaEventElapsedTime(&d2h_copy, start, end);

  // TODO: Adapt check_result() and invoke
  check_result(grid_ref, grid_opt3, N);

  std::cout << "Host to device Memcpy time (ms): "<< h2d_copy << endl;
  std::cout << "Pinned Memory Kernel time (ms): " << kernel_time << endl;
  std::cout << "Device to host Memcpy time (ms): "<< d2h_copy <<endl<<endl;
  cudaFree(d_grid_out);
  // =============================================================================================================================================================



  // TODO: Fill in kernel5 =======================================================================================================================================
  double *v_grid_in = nullptr, *v_grid_out = nullptr;
  cudaCheckError(cudaMallocManaged(&v_grid_in, sizeof(double)*SIZE));
  cudaCheckError(cudaMallocManaged(&v_grid_out, sizeof(double)*SIZE));

  for(int i=0;i<SIZE;i++) v_grid_in[i] = grid_in[i]; // so that all kernels receive the same input

  // s_mem_size = ((chunk+2)*(chunk+2)*(chunk+2))*sizeof(double); // shared memory size
  // dim3 blocksize(chunk, chunk, chunk);
  // dim3 gridsize(N/chunk, N/chunk, N/chunk);

  cudaEventRecord(start,0);
  kernel1<<<(SIZE+TPB-1/TPB), TPB>>>(v_grid_in, v_grid_out, N);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);

  // TODO: Adapt check_result() and invoke
  check_result(grid_ref, v_grid_out, N);

  std::cout << "UVM Kernel time (ms): " << kernel_time << "\n";
  // =============================================================================================================================================================

  // TODO: Free memory
  free(grid_in);
  free(grid_out);
  free(grid_opt1);
  free(grid_opt2);
  cudaFreeHost(grid_opt3);
  cudaFreeHost(h_grid_in);
  cudaFree(d_grid_in);
  cudaFree(d_grid_out);
  cudaFree(v_grid_in);
  cudaFree(v_grid_out);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return EXIT_SUCCESS;
}

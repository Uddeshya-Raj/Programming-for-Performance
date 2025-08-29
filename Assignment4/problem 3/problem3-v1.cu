#include <unistd.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm>

using namespace std;

#define NSEC_SEC_MUL (1.0e9) 

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

struct timespec begin_grid, end_main;

int pnts = 0;

__device__ double global_storage[12000][10];
__device__ int counter;

__device__ void recursive_func(double *grid, double *c, double *e, double *x, double *sums, int index){
  if(index >= 10){
    bool good_sum = true;
    for(int k=0;k<10;k++){
      if(fabs(sums[k]) > e[k]){
        good_sum = false;
        break;
      }
    }
    if(good_sum){
      int idx = atomicAdd(&counter,1);
      global_storage[idx][0] = x[0];
      global_storage[idx][1] = x[1];
      global_storage[idx][2] = x[2];
      global_storage[idx][3] = x[3];
      global_storage[idx][4] = x[4];
      global_storage[idx][5] = x[5];
      global_storage[idx][6] = x[6];
      global_storage[idx][7] = x[7];
      global_storage[idx][8] = x[8];
      global_storage[idx][9] = x[9];
    }
    return;
  }

  double start = grid[index*3+0];
  double end = grid[index*3+1];
  double step = grid[index*3+2];
  for(double xi=start; xi<end; xi+=step){
    double temp[10] = {0.0f};
    bool bad_i = false;
    for(int j=0;j<10;j++){
      temp[j] = sums[j]+c[j*10+index]*xi;
      if(temp[j]>e[j]){
        bad_i = true;
        break;
      }
    }
    if(bad_i) continue;
    else{
      double new_x[10];
      for(int i=0;i<10;i++) new_x[i]=x[i];
      new_x[index] = xi;
      recursive_func(grid, c, e, new_x, temp, index+1);
    }
  }
}

__global__ void wrapper_kernel(double *grid, double *c, double *e, double *x, double *sums, int index){

  if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
    
  }

  float x0 = grid[0*3+2]*blockIdx.x + grid[0*3+0]; //using x direction for mimicing 0th level of loop
  float x1 = grid[1*3+2]*blockIdx.y + grid[1*3+0]; //using y direction for mimicing 1st level of loop
  float x2 = grid[2*3+2]*blockIdx.z + grid[2*3+0]; //using z direction for mimicing 2nd level of loop
  // check if current 3 values of x are effective or not
  double temp[10] = {0.0f};
  bool bad_values = false;
  for(int i=0;i<10;i++){
    temp[i] = sums[i] + c[i*10+0]*x0 + c[i*10+1]*x1 + c[i*10+2]*x2;
    if(temp[i] > e[i]){
      bad_values = true;
      break;
    }
  }
  if(!bad_values){
    // int idx = atomicAdd(&counter, 1);
    double new_x[10] = {0.0f};
    for(int i=0;i<10;i++) new_x[i] = x[i];
    new_x[0] = x0;
    new_x[1] = x1;
    new_x[2] = x2;
    recursive_func(grid, c, e, new_x, temp, index+3);
  }
}

// original call : func(grid, c, e, x[0,0,0,...], sums[-d0,-d1,-d2,...], 0)
void func(double (*grid)[3], double (*c)[10], double *e, double *x, double *sums, int index, FILE* fptr){
  if(index >= 10){
    bool good_sum = true;
    for(int k=0;k<10;k++){
      if(fabs(sums[k]) > e[k]){
        good_sum = false;
        break;
      }
    }
    if(good_sum){
      pnts += 1;
      fprintf(fptr, "%lf\t", x[0]);
      fprintf(fptr, "%lf\t", x[1]);
      fprintf(fptr, "%lf\t", x[2]);
      fprintf(fptr, "%lf\t", x[3]);
      fprintf(fptr, "%lf\t", x[4]);
      fprintf(fptr, "%lf\t", x[5]);
      fprintf(fptr, "%lf\t", x[6]);
      fprintf(fptr, "%lf\t", x[7]);
      fprintf(fptr, "%lf\t", x[8]);
      fprintf(fptr, "%lf\n", x[9]);
    }
    return;
  }  
  double start = grid[index][0];
  double end = grid[index][1];
  double step = grid[index][2];
  for(double xi=start;xi<end;xi+=step){
    double temp[10] = {0.0f};
    bool bad_i = false;
    for(int j=0;j<10;j++){
      temp[j] = sums[j]+c[j][index]*xi;
      if(temp[j]>e[j]){
        bad_i = true;
        break;
      }
    }
    if(bad_i) continue;
    else{
      double new_x[10] = {0.0f};
      for(int i=0;i<10;i++) new_x[i] = x[i];
      new_x[index] = xi;
      func(grid, c, e, new_x, temp, index+1, fptr);
    }
  }
}

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {

  cudaError_t err;
  size_t newStackSize = 4096; // <--- adjusted stack size
  err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
  if (err != cudaSuccess) {
      std::cerr << "Error setting stack size: " << cudaGetErrorString(err) << std::endl;
      return -1;
  }

  size_t currentStackSize;
  err = cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize); 
  if (err != cudaSuccess) {
      std::cerr << "Error getting stack size limit: " << cudaGetErrorString(err) << std::endl;
      return -1;
  }
  std::cout << "New stack size limit: " << currentStackSize << " bytes" << std::endl;

  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  cudaMemset(&counter, 0, sizeof(int));
  
  double kk = 0.3;

  int pos = 0;
  double grid[10][3], *d_grid;
  for(int i=0;i<10;i++){
    for(int j=0;j<3;j++){
      grid[i][j] = b[pos++];
    }
  }

  pos = 0;
  double c[10][10], *d_c;
  double d[10];
  double ey[10];
  double e[10], d_e[10];

  for(int i = 0; i<10; i++){
    for(int j=0; j<10; j++){
      c[i][j] = a[pos++];
    }
    d[i] = a[pos++];
    ey[i] = a[pos++];
    e[i] = ey[i]*kk;
  }

  double x[10] = {0.0f}, d_x[10];
  double sums[10], d_sums[10];
  for(int i=0;i<10;i++) sums[i] = -d[i];

  cudaCheckError(cudaMalloc((void**)&d_grid, 30*sizeof(double)));
  cudaCheckError(cudaMemcpy(d_grid, grid, sizeof(double)*30, cudaMemcpyHostToDevice));

  cudaCheckError(cudaMalloc((void**)&d_c, 100*sizeof(double)));
  cudaCheckError(cudaMemcpy(d_c, c, sizeof(double)*100, cudaMemcpyHostToDevice));

  cudaCheckError(cudaMalloc((void**)&d_e, sizeof(double)*10));
  cudaCheckError(cudaMemcpy(d_e, e, sizeof(double)*10, cudaMemcpyHostToDevice));

  cudaCheckError(cudaMalloc((void**)&d_x, sizeof(double)*10));
  cudaCheckError(cudaMemcpy(d_x, x, sizeof(double)*10, cudaMemcpyHostToDevice));

  cudaCheckError(cudaMalloc((void**)&d_sums, sizeof(double)*10));
  cudaCheckError(cudaMemcpy(d_sums, sums, sizeof(double)*10, cudaMemcpyHostToDevice));


  FILE *fptr = fopen("./results-v1.txt", "w");
  if(fptr == NULL){
    printf("Error in creating file !");
    exit(1);
  }
  // clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  // func(grid, c, e, x, sums, 0, fptr);  
  // clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);

  dim3 blockpergird((grid[0][1]-grid[0][0])/grid[0][2], (grid[1][1]-grid[1][0])/grid[1][2], (grid[2][1]-grid[2][0])/grid[2][2]);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  // wrapper_kernel(double (*grid)[3], double (*c)[10], double *e, double *x, double *sums, int index)
  cudaEventRecord(start,0);
  wrapper_kernel<<<blockpergird, 1>>>(d_grid, d_c, d_e, d_x, d_sums, 0);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time,start,end);
  // cudaCheckError(cudaPeekAtLastError());
  // cudaCheckError(cudaDeviceSynchronize());

  int h_counter = 0;
  double (*output)[10];
  output = (double(*)[10])malloc(12000 * 10 * sizeof(double));
  cudaCheckError(cudaMemcpyFromSymbol(&h_counter, counter, sizeof(int), 0, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpyFromSymbol(output, global_storage, sizeof(double)*h_counter*10, 0, cudaMemcpyDeviceToHost));

  vector<vector<double>> data(h_counter, vector<double>(10));
  for (int i = 0; i < h_counter; i++) {
      for (int j = 0; j < 10; j++) {
          data[i][j] = output[i][j];
      }
  }

  std::sort(data.begin(), data.end());

  for (const auto& row : data) {
      for (const auto& value : row) {
          fprintf(fptr, "%lf\t", value);
      }
      fprintf(fptr, "\n");
  }
  fclose(fptr);

  // printf("result pnts (CPU) : %d \n", pnts);
  printf("result pnts (GPU) : %d \n", h_counter);
  // printf("cpu time = %f seconds\n",
  //        (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
  //            (end_main.tv_sec - begin_grid.tv_sec));
  printf("kernel time = %f seconds\n",kernel_time/1000);
  return EXIT_SUCCESS;
}


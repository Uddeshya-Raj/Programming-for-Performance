// #!bash
// export PAPI_DIR=<you location where PAPI is installed>
// export PATH=${PAPI_DIR}/bin:$PATH
// export LD_LIBRARY_PATH=${PAPI_DIR}/lib:$LD_LIBRARY_PATH
// PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"

#include <cassert>
#include <chrono>
#include <iostream>
#include <math.h>
#include <papi.h>

using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)

void handle_error(int retval) {
  cout << "PAPI error: " << retval << ": " << PAPI_strerror(retval) << "\n";
  exit(EXIT_FAILURE);
}

void matmul_ijk(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

// void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
//   for (int i = 0; i < SIZE; i++) {
//     for (int j = 0; j < SIZE; j++) {
//       for (int k = 0; k < SIZE; k++) {
//         C[i*SIZE + k] += A[i * SIZE + j] * B[j * SIZE + k];
//       }
//     }
//   }
// }

void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE, int* blk_size) {
  for(int i = 0; i < SIZE; i += blk_size[0])
    for(int j = 0; j < SIZE; j += blk_size[1])
      for(int k = 0; k < SIZE; k += blk_size[2])


        for(int ii = i; ii < min(N, i+blk_size[0]); ii++)
          for(int jj = j; jj < min(N, j+blk_size[1]); jj++)
            for(int kk = k; kk < min(N, k+blk_size[2]); kk++)
              C[ii*SIZE + jj] += A[ii*SIZE + kk]*B[kk*SIZE + jj];
}



void init(uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      mat[i * SIZE + j] = 1;
    }
  }
}

void print_matrix(const uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << mat[i * SIZE + j] << "\t";
    }
    cout << "\n";
  }
}

void check_result(const uint32_t *ref, const uint32_t *opt, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (ref[i * SIZE + j] != opt[i * SIZE + j]) {
        assert(false && "Diff found between sequential and blocked versions!\n");
      }
    }
  }
}

int main() {
  uint32_t *A = new uint32_t[N * N];
  uint32_t *B = new uint32_t[N * N];
  uint32_t *C_seq = new uint32_t[N * N];
  uint32_t *C_blk = new uint32_t[N * N];

  //initializing PAPI library
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if(retval != PAPI_VER_CURRENT){
    fprintf(stderr,"PAPI Library init error\n");
    exit(1);
  }

    
    
  init(A, N);
  init(B, N);
  long long values[1];
  long long seq_miss, blk_miss;
  int EventSet = PAPI_NULL;

  int status = PAPI_create_eventset(&EventSet);
  if (status != PAPI_OK){
      fprintf(stderr, "Error creating event set\n");
      handle_error(status);
  }
  
  status = PAPI_add_event(EventSet, PAPI_L2_DCM);
  if (status != PAPI_OK){
      fprintf(stderr, "Error adding L2 D-Cache miss event to event set\n");
      handle_error(status);
  } 

  cout<<"blk size \t || time (seq) \t || time (BLK) \t || speedup \t || L2 D-cache misses saved \t "<<endl;
  cout<<"-------------------------------------------------------------------------------------------------"<<endl;
  // for(int i=1;i<=6;i++){

    // uint32_t blk = pow(2,i);
    int blk[3] = {8,8,32};
    for(int j=0;j<5;j++){

      init(C_seq, N);
      init(C_blk, N);

      cout<<" "<<blk[0]<<", "<<blk[1]<<", "<<blk[2]<<"\t ||";

      status = PAPI_start(EventSet);
      if(status != PAPI_OK){
          fprintf(stderr,"Error starting event counter\n");
          handle_error(status);
      }

      HRTimer start = HR::now();
      matmul_ijk(A, B, C_seq, N);
      HRTimer end = HR::now();

      if(PAPI_stop(EventSet, values) != PAPI_OK){
          fprintf(stderr,"Error stoping event counter\n");
          exit(1);
      }

      if(PAPI_read(EventSet, values) != PAPI_OK){
          fprintf(stderr,"Error reading event counters\n");
          exit(1);
      }

      seq_miss = values[0];

      auto duration_seq = duration_cast<microseconds>(end - start).count();
      // cout << "Time without blocking (us): " << duration << "\n";
      cout<<" "<<duration_seq<<" \t ||";

      status = PAPI_start(EventSet);
      if(status != PAPI_OK){
          fprintf(stderr,"Error starting event counter\n");
          handle_error(status);
      }
      
      start = HR::now();
      matmul_ijk_blocking(A, B, C_blk, N, blk);
      end = HR::now();

      if(PAPI_stop(EventSet, values) != PAPI_OK){
          fprintf(stderr,"Error stoping event counter\n");
          exit(1);
      }

      if(PAPI_read(EventSet, values) != PAPI_OK){
          fprintf(stderr,"Error reading event counters\n");
          exit(1);
      }

      blk_miss = values[0];

      auto duration_blk = duration_cast<microseconds>(end - start).count();
      // cout << "Time with blocking (us): " << duration << "\n";
      cout<<" "<<duration_blk<<" \t ||";
      printf(" %0.2fx \t ||",(double)(duration_seq)/(double)(duration_blk));
      cout<<" "<<seq_miss-blk_miss<<" \t "<<endl;

      // cout<<"C1"<<endl;
      // print_matrix(C_seq,N);
      // cout<<"C2"<<endl;
      // print_matrix(C_blk,N);
      check_result(C_seq, C_blk, N);


    }
    cout<<"----------------------------------------------------------------------"<<endl;
  // } 

  return EXIT_SUCCESS;
}

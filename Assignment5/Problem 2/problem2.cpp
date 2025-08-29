#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>
#include "stack.hpp"

using std::cout;
using std::endl;
using std::string;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::filesystem::path;

static constexpr uint64_t RANDOM_SEED = 42;
static const uint32_t bucket_count = 1000;
static constexpr uint64_t MAX_OPERATIONS = 1e+15;
static const uint32_t SENTINEL_KEY = 0;
static const uint32_t SENTINEL_VALUE = 0;
static const uint32_t PROBING_RETRIES = (1 << 20);
static const uint32_t TOMBSTONE_KEY = UINT32_MAX;

int NUM_THREADS = 1;

// Pack key-value into a 64-bit integer
inline uint64_t packKeyValue(uint32_t key, uint32_t val) {
  return (static_cast<uint64_t>(key) << 32) |
         (static_cast<uint32_t>(val) & 0xFFFFFFFF);
}

// Function to unpack a 64-bit integer into two 32-bit integers
inline void unpackKeyValue(uint64_t value, uint32_t& key, uint32_t& val) {
  key = static_cast<uint32_t>(value >> 32);
  val = static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void create_file(path pth, const uint32_t* data, uint64_t size) {
  FILE* fptr = NULL;
  fptr = fopen(pth.string().c_str(), "wb+");
  fwrite(data, sizeof(uint32_t), size, fptr);
  fclose(fptr);
}

/** Read n integer data from file given by pth and fill in the output variable
    data */
void read_data(path pth, uint64_t n, uint32_t* data) {
  FILE* fptr = fopen(pth.string().c_str(), "rb");
  string fname = pth.string();
  if (!fptr) {
    string error_msg = "Unable to open file: " + fname;
    perror(error_msg.c_str());
  }
  int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
  if (freadStatus == 0) {
    string error_string = "Unable to read the file " + fname;
    perror(error_string.c_str());
  }
  fclose(fptr);
}

// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1e5;
/** percentage of insert queries */
float POP_probability = 0.3f;
/** number of iterations */
uint64_t runs = 2;

// List of valid flags and description
void validFlagsDescription() {
  cout << "ops: specify total number of operations\n";
  cout << "rns: the number of iterations\n";
  cout << "pop: probability that a random operation will be pop instead of push\n";
  cout << "not using an option will cause it have default value defined in code\n";
}

// Code snippet to parse command line flags and initialize the variables
int parse_args(char* arg) {
  string s = string(arg);
  string s1;
  uint64_t val;

  try {
    s1 = s.substr(0, 4);
    string s2 = s.substr(5);
    val = stol(s2);
  } catch (...) {
    cout << "Supported: " << std::endl;
    cout << "-*=[], where * is:" << std::endl;
    validFlagsDescription();
    return 1;
  }

  if (s1 == "-ops") {
    NUM_OPS = val;
  } else if (s1 == "-rns") {
    runs = val;
  } else if (s1 == "-pop") {
    POP_probability = val;
  } else {
    std::cout << "Unsupported flag:" << s1 << "\n";
    std::cout << "Use the below list flags:\n";
    validFlagsDescription();
    return 1;
  }
  return 0;
}

void stack_testing(concurrent_stack* mystack, int ops, int num_threads, float pop_prob, uint32_t* arr) {
  // Seed for random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Parallel region using OpenMP
  #pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    for (int i = 0; i < ops/num_threads; ++i) {
        float rand_val = dis(gen); 

      if (rand_val < pop_prob) {
        int popped_value = mystack->pop();
        // cout<<"pop --> "<<popped_value<<" \t\t";
      } else {
        mystack->push(arr[i]);
        // cout<<"push <-- "<<arr[i]<<" \t\t";
      }
    }
  }
}

int main(int argc, char* argv[]) {
  srand(time(0));

  for (int i = 1; i < argc; i++) {
    int error = parse_args(argv[i]);
    if (error == 1) {
      cout << "Argument error, terminating run.\n";
      exit(EXIT_FAILURE);
    }
  }

  cout << "NUM OPS: " << NUM_OPS << " \nProbability of a random operation being POP instead of PUSH: " << POP_probability << "\n";

  auto* h_values_push = new uint32_t[NUM_OPS];
  memset(h_values_push, 0, sizeof(uint32_t) * NUM_OPS);

  // Use shared files filled with random numbers
  path cwd = std::filesystem::current_path();
  path path_push_values = cwd / "random_values_insert.bin";

  assert(std::filesystem::exists(path_push_values));

  // Read data from file
  auto* tmp_values_push = new uint32_t[NUM_OPS];
  read_data(path_push_values, NUM_OPS, tmp_values_push);
  for (int i = 0; i < NUM_OPS; i++) {
    h_values_push[i] = tmp_values_push[i];
  }
  delete[] tmp_values_push;

  // Max limit of the uint32_t: 4,294,967,295
  std::mt19937 gen(RANDOM_SEED);
  std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

  float total_run_time = 0.0F;

  HRTimer start, end;
  for (int i = 0; i < runs; i++) {
    concurrent_stack* test_stack = new concurrent_stack();
    start = HR::now();
    stack_testing(test_stack,NUM_OPS, NUM_THREADS, POP_probability, h_values_push);
    end = HR::now();
    float iter_run_time = duration_cast<milliseconds>(end - start).count();
    cout<<"run "<<i+1<<" time(ms) : "<<iter_run_time<<endl;
    cout<<"\n";
    total_run_time += iter_run_time;
    delete test_stack;
  }


  cout << "Avg time taken by push and pop operations (ms): " << total_run_time / runs
       << "\n";

  return EXIT_SUCCESS;
}

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <pthread.h>
#include <string>
#include <bits/stdc++.h>
#include <fstream>
#include <omp.h>

using namespace std;
using std::atomic_int;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::list;
using std::ofstream;
using std::strerror;
using std::string;

int thread_count;
int shared_buffer_max_size; //in lines
int lines_to_be_read;
string input_path;
string output_path;
fstream input_file;
fstream output_file;

queue<string> shared_buffer;

struct thread_info{
    bool exists = false;
    pid_t tid = 0; 
} semicomplete_thread;

pthread_mutex_t shared_var_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t produce = PTHREAD_COND_INITIALIZER;
pthread_cond_t consume = PTHREAD_COND_INITIALIZER;
pthread_cond_t req_thread = PTHREAD_COND_INITIALIZER;
omp_lock_t buffer_lock;

void producer(int tid){

    string line;
    cout<<"inside thread : "<<tid<<endl;
    while(!input_file.eof()){
        
        omp_set_lock(&buffer_lock);

        cout<<"inside critical section: "<<tid<<endl;

        pthread_mutex_lock(&shared_var_mutex); //this mutex is just used to implement cond_vars. cond_var will not work with omp_lock_t.
        cout<<"dummy lock aquired "<< tid<<endl;
                
        bool var = true;
        while(var){
            if(semicomplete_thread.exists && semicomplete_thread.tid != tid){
                //release shared_var_mutex lock and get blocked
                cout<<tid<<" not the correct thread. going to sleep."<<endl;
                pthread_cond_wait(&req_thread, &shared_var_mutex);
            }
            else {
                var = false;
            }
        }

        assert(!semicomplete_thread.exists || semicomplete_thread.tid == tid);
        semicomplete_thread.exists = true;
        semicomplete_thread.tid = tid;

        while(shared_buffer_max_size - shared_buffer.size() == 0){
            // wake up consumer
            cout<<"Elements in buffer: "<<shared_buffer.size()<<endl;
            pthread_cond_signal(&consume);
            // producer lock released, goes to sleep
            pthread_cond_wait(&produce, &shared_var_mutex);
            // #pragma omp taskyield
            cout<<"producer "<<tid<<" resumed"<<endl;
        }

        int L = lines_to_be_read;

        while(!input_file.eof() && L--){
            if(shared_buffer_max_size == shared_buffer.size()){
                // wake up consumer
                pthread_cond_signal(&consume);
                cout<<"need to clear buffer. consumer called. "<<tid<<" going to sleep."<<endl;
                // block producer
                pthread_cond_wait(&req_thread, &shared_var_mutex); 
                // #pragma omp taskyield
                cout<<"thread "<<tid<<" resumed to complete the job"<<endl;
            }
            getline(input_file, line);

            // uncomment this line to see which line was written by which thread
            // line  = "("+to_string(tid)+") "+line;
            
            shared_buffer.push(line);
            cout<<"producer "<<tid<<" pushed to buffer : "<<line<<endl;
        }
        semicomplete_thread.tid = 0;
        semicomplete_thread.exists = false;
        

        pthread_mutex_unlock(&shared_var_mutex);

        omp_unset_lock(&buffer_lock);
        cout<<tid<<" released buffer lock"<<endl;
        // wake all producer threads
        pthread_cond_broadcast(&req_thread);
        // wake consumer threads
        pthread_cond_signal(&consume);
    }        
    
}

void consumer(){

    while(!input_file.eof() || !shared_buffer.empty()){
        #pragma omp critical
        {
            pthread_mutex_lock(&shared_var_mutex);
            if(shared_buffer.empty()){
                cout<<"Shared buffer empty"<<endl;
                // signal producer threads
                pthread_cond_signal(&produce);
                // block consumer
                pthread_cond_wait(&consume, &shared_var_mutex);
                // #pragma omp taskyield
            }
            output_file<<shared_buffer.front()<<endl;
            cout<<"consumer wrote to file : "<<shared_buffer.front()<<endl;
            shared_buffer.pop();
            pthread_mutex_unlock(&shared_var_mutex);
        }
        //signal all producer threads
        pthread_cond_broadcast(&produce);
        pthread_cond_broadcast(&req_thread);
    }
    
}

int main(int argc, char *argv[]) {
    if(argc != 6){ 
        cerr << "Usage: " << argv[0] << " -inp=<input_file> -thr=<thread count> -lns=<lines_to_be_read> -buf=<buffer_size> -out=<output_file>" << std::endl;
        exit(1);
    }

    std::map<string, string> parameters;

    for(int i=1; i<argc; i++){
        string args(argv[i]);
        string p_name = args.substr(0,args.find('='));
        string p_value = args.substr(args.find('=')+1);
        parameters[p_name] = p_value;
    }

    // for(auto itr = parameters.begin(); itr != parameters.end(); itr++){
    //     cout<<itr->first<<" = "<<itr->second<<endl;
    // }

    input_path = parameters["-inp"];
    thread_count = stol(parameters["-thr"],NULL,10);
    lines_to_be_read = stol(parameters["-lns"],NULL,10);
    shared_buffer_max_size = stol(parameters["-buf"],NULL,10);
    output_path = parameters["-out"];

    if(thread_count*lines_to_be_read*shared_buffer_max_size == 0) return 0; 

    input_file.open(input_path, ios::in);
    if(!input_file.is_open()){
        cerr<<"Error opening input file"<<endl;
        exit(EXIT_FAILURE);
    }

    output_file.open(output_path, ios::out|ios::ate|ios::trunc);
    if(!output_file.is_open()){
        cerr<<"Error opening output file"<<endl;
        exit(EXIT_FAILURE);
    }

    //-------------------------------------------------

    omp_init_lock(&buffer_lock);

    #pragma omp parallel num_threads(thread_count+1)
    {
        int tid = omp_get_thread_num();
        if(tid < thread_count){
            producer(tid);
        } else {
            consumer();
        }
    }
    
   omp_destroy_lock(&buffer_lock);


    input_file.close();
    output_file.close();

    return EXIT_SUCCESS; 
} 

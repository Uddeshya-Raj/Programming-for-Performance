#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <bits/stdc++.h>
#include <pthread.h>

using namespace std;

bool job_done = false;
int thread_count;
int shared_buffer_max_size; //in lines
int lines_to_be_read;
string input_path;
string output_path;
fstream input_file;
fstream output_file;

queue<string> shared_buffer;
int threads_completed = 0;

struct thread_info{
    bool exists = false;
    pid_t tid = 0;
} semicomplete_thread;

pthread_mutex_t shared_var_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t produce = PTHREAD_COND_INITIALIZER;
pthread_cond_t consume = PTHREAD_COND_INITIALIZER;
pthread_cond_t req_thread = PTHREAD_COND_INITIALIZER;
pthread_barrier_t barrier;



void* fill_shared_buffer(void *th_arg){
    string line;


    //all threads should be created before they contend for mutex lock
    pthread_barrier_wait(&barrier);
    // cout<<gettid()<<" Crossed the barrier"<<endl;
    pthread_mutex_lock(&shared_var_mutex);

    bool var = true;
    while(var){
        if(semicomplete_thread.exists && semicomplete_thread.tid != gettid())
            pthread_cond_wait(&req_thread, &shared_var_mutex);
        else var = false;
    }

    // cout<<"buffer lock aquired by "<<gettid()<<endl;
    // cout<<gettid()<<" in producer thread"<<endl;

    //if not enough space for L reads
    while(shared_buffer_max_size - shared_buffer.size() == 0){
        // cout<<"Elements in buffer: "<<shared_buffer.size()<<endl;        
        pthread_cond_signal(&consume);
        pthread_cond_wait(&produce, &shared_var_mutex);
        // cout<<"producer "<<gettid()<<" resumed"<<endl;
    }
    int L = lines_to_be_read;
    while(!input_file.eof()&&L--){
        assert(!semicomplete_thread.exists || semicomplete_thread.tid == gettid());
        if(shared_buffer.size() == shared_buffer_max_size){
            semicomplete_thread.exists = true;
            semicomplete_thread.tid = gettid();
            pthread_cond_signal(&consume);
            pthread_cond_wait(&req_thread, &shared_var_mutex);
            // cout<<"thread "<<gettid()<<" resumed to complete the job"<<endl;
        }
        getline(input_file, line);
        shared_buffer.push(line);
    }

    semicomplete_thread.tid = 0;
    semicomplete_thread.exists = false;

    threads_completed++;
    if(threads_completed == thread_count) job_done=true;

    // cout<<gettid()<<" releasing buffer lock"<<endl;
    pthread_mutex_unlock(&shared_var_mutex);
    pthread_cond_broadcast(&req_thread);
    pthread_cond_signal(&consume);


    pthread_exit(nullptr);
}

void* consume_shared_buffer(void *th_arg){
    while(!job_done || !shared_buffer.empty()){
        pthread_mutex_lock(&shared_var_mutex);

        //if nothing to write from buffer
        if(shared_buffer.empty()){
            // cout<<"Shared buffer empty"<<endl;
            pthread_cond_signal(&produce);
            pthread_cond_wait(&consume, &shared_var_mutex);
        }
        output_file<<shared_buffer.front()<<endl;
        // cout<<"consumer wrote to file : "<<shared_buffer.front()<<endl;
        shared_buffer.pop();
        pthread_mutex_unlock(&shared_var_mutex);
        pthread_cond_broadcast(&produce);
        pthread_cond_broadcast(&req_thread);
    }
    pthread_exit(nullptr);
}

int main(int argc, char *argv[]){
    if(argc != 6){ 
        cerr << "Usage: " << argv[0] << " <input_file> <num_threads> <lines_per_thread> <buffer_size> <output_file>" << std::endl;
        exit(1);
    }

    map<string, string> parameters;

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
    
    // cout<<"BUFFER CREATED"<<endl;

    pthread_barrier_init(&barrier,NULL,thread_count);

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
    
    // cout<<"INPUT and Output files opened"<<endl;

    pthread_t producer_threads[thread_count];
    pthread_t consumer_thread;
    
    pthread_create(&consumer_thread,NULL,consume_shared_buffer,NULL);
    for(int i=0;i<thread_count;i++){
        pthread_create(&producer_threads[i], NULL, fill_shared_buffer, NULL);
        // cout<<"Thread "<<i+1<<" created"<<endl;
    }

    for(int i=0;i<thread_count;i++){
        pthread_join(producer_threads[i],NULL);
    }
    pthread_join(consumer_thread,NULL);

    pthread_barrier_destroy(&barrier);
    input_file.close();
    output_file.close();

    return 0;
}
#ifndef STACK_HPP
#define STACK_HPP

#include <unistd.h>
#include <iostream>
#include <atomic>
#include <climits>
#include <thread>
#include <chrono>
#include <vector>

#endif // STACK_HPP

using namespace std;


struct Node
{
    int value;
    Node* next;
    Node(int input) : value(input), next(nullptr) {};
};

class concurrent_stack
{
private:
    Node* top = nullptr;
    atomic<uint8_t> tag = 0; // will only have values 0 to 15
    uint32_t max_time = 20, min_time = 1; // time in milliseconds
    uint32_t random_seed = time(0);
    // vector<Node*> used_addresses;
protected:
    bool tryPush(Node* node){
        uint8_t old_tag = tag.load();
        Node* old_top = top;
        node->next = old_top;
        old_top = reinterpret_cast<Node*>(reinterpret_cast<uintptr_t>(old_top) | old_tag-tag.load());
        return(__atomic_compare_exchange_n(&top, &old_top, node, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
    int tryPop(){
        uint8_t old_tag = tag.load();
        if(top == nullptr) return -1;
        Node* old_top = top;
        Node* new_top = old_top->next;
        old_top = reinterpret_cast<Node*>(reinterpret_cast<uintptr_t>(old_top) | old_tag-tag.load());
        if(__atomic_compare_exchange_n(&top, &old_top, new_top, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){
            int val = old_top->value;
            delete old_top;
            return val;
        }else{
            return 0;
        }
    }
public:
    concurrent_stack(){
        srand(random_seed);
    };
    ~concurrent_stack(){
        // for(auto pointer: used_addresses){
        //     delete(pointer);
        // }
    };
    void push(int v){
        Node* new_node = new Node(v);
        // used_addresses.push_back(new_node);
        while(true){
            if(tryPush(new_node)) {
                tag = (tag+1)%16;
                return;
            }
            else{
                uint32_t backoff = min_time + static_cast<double>(rand()) / (RAND_MAX / (max_time - min_time));
                this_thread::sleep_for(chrono::milliseconds(backoff));
            }
        }
    }
    int pop(){
        while(true){
            int retval = tryPop();
            if(retval) {
                if(retval != -1) tag = (tag+1)%16;
                return retval;
            }
            else{
                uint32_t backoff = min_time + static_cast<double>(rand()) / (RAND_MAX / (max_time - min_time));
                this_thread::sleep_for(chrono::milliseconds(backoff));
            }
        }
    }
    void print(){
        Node* curr = top;
        cout<<"TOP ";
        while(curr!=nullptr) cout<<"-->"<<curr->value;
        cout<<"--> nullptr"<<endl;
    }
};

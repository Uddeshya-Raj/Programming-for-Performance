#ifndef HASH_HPP
#define HASH_HPP

#include <vector>
#include <array>
#include <mutex>
#include <functional>
#include <cstdint>
#include <utility>
#include <nmmintrin.h>
#include <iostream>

#endif // HASH_HPP

using namespace std;

const size_t PROBE_SIZE = 8;
const int LIMIT = 5;
const int THRESHOLD = 1;
class PhasedCuckooHashTable {
private:
    int capacity;
    int size;
    int og_lock_len = 0;
    std::vector<std::vector<std::vector<std::pair<int,int>>>> table;
    std::recursive_mutex** locks = new std::recursive_mutex*[2]; // locks

protected:
    // Acquire lock for the specific bucket
    void acquire(int tableIndex, int hashIndex) {
        locks[tableIndex][hashIndex].lock();
        // cout<<"locking table ["<<tableIndex<<"]["<<hashIndex<<"]"<<endl;
    }

    // Release lock for the specific bucket
    void release(int tableIndex, int hashIndex) {
        locks[tableIndex][hashIndex].unlock();
        // cout<<"unlocking table ["<<tableIndex<<"]["<<hashIndex<<"]"<<endl;
    }

    std::pair<std::unique_lock<std::recursive_mutex>, std::unique_lock<std::recursive_mutex>> acquire(const int& x) {
        int h0 = hash0(x) % og_lock_len;
        int h1 = hash1(x) % og_lock_len;

        // Lock both mutexes
        std::unique_lock<std::recursive_mutex> lock1(locks[0][h0], std::defer_lock);
        std::unique_lock<std::recursive_mutex> lock2(locks[1][h1], std::defer_lock);
        std::lock(lock1, lock2); // Lock them safely without deadlocks
        // std::cout<<"locking table["<<0<<"]["<<h0<<"] and table["<<1<<"]["<<h1<<"]"<<std::endl;
        return {std::move(lock1), std::move(lock2)};
    }

    void release(std::pair<std::unique_lock<std::recursive_mutex>, std::unique_lock<std::recursive_mutex>>& locks) {
        locks.first.unlock();
        locks.second.unlock();
        // std::cout<<"unlocked table"<<std::endl;
    }

    // Relocate method
    bool relocate(int i, int hi) {
        int hj = 0;
        int j = 1 - i;
        for (int round = 0; round < LIMIT; round++) {
            // cout<<"relocate table["<<i<<"]["<<hi<<"]"<<endl;
            auto& iSet = table[i][hi];
            std::pair<int, int> y = iSet.front();
            int key = y.first;
            int value = y.second;

            switch (i) {
                case 0: hj = hash1(key) % capacity; break;
                case 1: hj = hash0(key) % capacity; break;
            }

            auto lock = acquire(key);

            auto& jSet = table[j][hj];

            try {
                // Remove y from iSet
                bool removed = false;
                for (auto it = iSet.begin(); it != iSet.end(); it++) {
                    if (*it == y) {
                        iSet.erase(it); // Remove y
                        removed = true;
                        // cout<<"removed "<<y.first<<","<<y.second<<" from table["<<i<<"]["<<hi<<"]"<<endl;
                        break;
                    }
                }

                if (removed) {
                    if (jSet.size() < THRESHOLD) {
                        jSet.push_back(y);
                        release(lock);
                        // cout<<"put "<<y.first<<","<<y.second<<" in table "<<j<<","<<hj<<endl;
                        return true;
                    } else if (jSet.size() < PROBE_SIZE) {
                        jSet.push_back(y);
                        // cout<<"would relocate "<<y.first<<","<<y.second<<" in table "<<j<<","<<hj<<endl;
                        i = 1 - i;
                        hi = hj;
                        j = 1 - j;

                    } else {
                        // Put y back
                        iSet.push_back(y);
                        release(lock);
                        // cout<<"put "<<y.first<<","<<y.second<<" back"<<endl;
                        return false;
                    }
                } else if (iSet.size() > THRESHOLD) {
                    release(lock);
                    continue;
                } else {
                    release(lock);
                    return true;
                }
            } catch (...) {
                release(lock);
                throw;
            }

            release(lock);
        }
        return false;
    }

    void resize(){
        int oldCapacity = capacity;
        for(int i=0;i<og_lock_len;i++){
            locks[0][i].lock();
        }
        try {
            if(capacity != oldCapacity) return;

            auto oldTable = table;
            capacity = 2*capacity;
            table = std::vector<std::vector<std::vector<std::pair<int,int>>>>();
            table.resize(2, std::vector<std::vector<std::pair<int, int>>>(capacity));
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < capacity; ++j) {
                    table[i][j] = std::vector<std::pair<int, int>>(); // Initialize each probe set with default values
                }
            }

            for(auto& row: oldTable){
                for(auto& set: row){
                    for(auto& p: set){
                        // std::cout<<" rehashing ";  
                        // std::cout<<p.first<<','<<p.second<<'\n';                        
                        add(p.first, p.second);
                    }
                }
            }

        }catch(...){
            for(int i=0;i<og_lock_len;i++){
                locks[0][i].unlock();
            }
            throw;
        }
        for(int i=0;i<og_lock_len;i++){
            locks[0][i].unlock();
        }
    }

    // Murmur Hash
    uint32_t hash0(const int& x) const {
        uint32_t h = std::hash<int>{}(x);
        h *= 0x5BD1E995;
        h ^= h >> 15;
        h *= 0x865C0D69;
        return h;    
    }

    uint32_t hash1(const int& x) const {
        uint32_t h = std::hash<int>{}(x);
        h += 0x9E3779B9;
        h += (h << 3);
        h ^= (h >> 11);
        h += (h << 15);
        return h;
    }

public:
    // Constructor
    PhasedCuckooHashTable(int set_size) : capacity(set_size) {
        size = 0;
        og_lock_len = capacity;
        table.resize(2, std::vector<std::vector<std::pair<int, int>>>(capacity));
        for(int i=0;i<2;i++){
            locks[i] = new std::recursive_mutex[capacity];
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < capacity; ++j) {
                table[i][j] = std::vector<std::pair<int, int>>(); // Initialize each probe set with default values
            }
        }
    }

    float load_factor(){
        return (double)(size)/(capacity*2);
    }


    // contains method
    bool contains(const int& key) {
        // cout<<"inside contains"<<endl;
        auto lock = acquire(key);
        int hashIndex0 = hash0(key) % capacity;
        int hashIndex1 = hash1(key) % capacity;
        // cout<<hashIndex0<<endl;
        // cout<<hashIndex1<<endl;
        try {
            // cout<<"table 0,"<<hashIndex0<<" size : "<<table[0][hashIndex0].size()<<endl;
            if(table[0][hashIndex0].size()){
                auto& set0 = table[0][hashIndex0];
                for (auto& elem : set0) {
                    if (elem.first == key) {
                        release(lock);
                        // std::cout<<"t0\n";
                        return true;
                    }
                }
            }
        } catch (...) {
            release(lock);
            throw; // Ensure we release the lock in case of an exception
        }

        try {
            // cout<<"table 1,"<<hashIndex1<<" size : "<<table[0][hashIndex0].size()<<endl;
            if(table[1][hashIndex1].size()){
                auto& set1 = table[1][hashIndex1];
                for (auto& elem : set1) {
                    if (elem.first == key) {
                        release(lock);
                        // std::cout<<"t1\n";
                        return true;
                    }
                }
            }
        } catch (...) {
            release(lock);
            throw; // Ensure we release the lock in case of an exception
        }
        release(lock);
        // cout<<"not found"<<endl;
        return false; // Item not found in either probe set
    }


    // Add method
    bool add(const int& key, const int& value) {
            // std::cout<<"key : "<<key<<" value : "<<value<<std::endl;
        auto lock = acquire(key);

        int h0 = hash0(key) % capacity;
        int h1 = hash1(key) % capacity;
        int i = -1, h = -1;
        bool mustResize = false;

        auto& set0 = table[0][h0];
        auto& set1 = table[1][h1];

        bool data_processed = true;
        do{
            if (contains(key)) {
                release(lock);
                // cout<<"already present"<<endl;
                return false;
            }

            if (set0.size() < THRESHOLD) {
                set0.push_back({key, value});
                release(lock);
                // cout<<"placed in "<<0<<','<<h0<<endl;
                size++;
                if(load_factor()>0.8) resize();
                return true;
            } else if (set1.size() < THRESHOLD) {
                set1.push_back({key, value});
                release(lock);
                // cout<<"placed in "<<1<<','<<h1<<endl;
                size++;
                if(load_factor()>0.8) resize();
                return true;
            } else if (set0.size() < PROBE_SIZE) {
                set0.push_back({key, value});
                // cout<<"relocated in "<<0<<","<<h0<<endl;
                size++;
                i = 0; h = h0;
            } else if (set1.size() < PROBE_SIZE) {
                set1.push_back({key, value});
                // cout<<"relocated in "<<1<<","<<h1<<endl;
                size++;
                i = 0; h = h0;
            } else {
                // cout<<"could not put "<<key<<','<<value<<" anywhere"<<endl;
                data_processed = false;
                mustResize = true;
            }
            release(lock);

            if (mustResize) {
                resize();
            } else if (!relocate(i, h)) {
                resize();
            }
            if(load_factor()>0.8) resize();
        }while(!data_processed);
        return true; // x must have been added
    }



    // Remove method
    bool remove(const int& key) {
        int hashIndex0 = hash0(key) % capacity;
        int hashIndex1 = hash1(key) % capacity;

        auto lock = acquire(key);
        try {
            if(table[0][hashIndex0].size()){
                auto& set0 = table[0][hashIndex0];
                for (auto it = set0.begin(); it != set0.end(); ++it) {
                    std::pair<int, int> p = *it;
                    if (p.first == key) {
                        set0.erase(it); // Remove the item
                        release(lock);
                        size--;
                        return true;
                    }
                }
            }
        } catch (...) {
            release(lock);
            throw; // Ensure we release the lock in case of an exception
        }

        try {
            if(table[1][hashIndex1].size()){
                auto& set1 = table[1][hashIndex1];
                const auto end1 = set1.end();
                for (auto it = set1.begin(); it != end1; ++it) {
                    std::pair<int, int> p = *it;
                    if (p.first == key) {
                        set1.erase(it); // Remove the item
                        release(lock);
                        size--;
                        return true;
                    }
                }
            }
        } catch (...) {
            release(lock);
            throw; // Ensure we release the lock in case of an exception
        }
        release(lock);
        return false; // Item not found in either probe set
    }

    

};


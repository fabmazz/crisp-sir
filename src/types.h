#ifndef EPITYPES_H
#define EPITYPES_H
#include <vector>
#include <map>
#include <valarray>
#include <iostream>



namespace Epi{
    typedef double real_t;
    typedef std::valarray<real_t> ArrayReal;
    typedef std::map<int, int> intdict;

    struct Node
    {
        const int idx;
        std::map<int,int> neighs;

        std::vector<ArrayReal> loglambdas;



        void add_neighbor(const int& idx, const int& T);

        Node(int i): idx(i) {};

    };

    //void Node::set_data();
    
    
    void Node::add_neighbor(const int& i, const int& T){

        if (i == this->idx){
            std::cerr << "Cannot add self egde"<<std::endl;
            throw std::runtime_error("Self edges not permitted");
        }
        int n = loglambdas.size();
        // n is the index of the new neighbor
        neighs.emplace(i, n);

        loglambdas.push_back(ArrayReal(0., T));

    }
}

#endif
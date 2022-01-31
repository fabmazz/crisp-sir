#ifndef EPITYPES_H
#define EPITYPES_H
#include <vector>
#include <map>
#include <valarray>
#include <iostream>

typedef double real_t;
typedef std::valarray<real_t> ArrayReal;
typedef std::map<int, int> intdict;


namespace Epi{

    struct Node
    {
        const int idx;
        uint T;
        std::map<int,int> neighs;

        std::vector<ArrayReal> loglambdas;



        void add_neighbor(const int& idx);
        bool add_contact(const int& j,  const real_t& loglam, const uint& t);
        //this actually sets the contact
        bool set_contact(const int& j, const real_t& loglam, const uint& t);

        Node(const int& i, const uint& T): idx(i) {
            this->T = T;
        };

        bool has_neigh(const int& j);

    };    
    
    void Node::add_neighbor(const int& i){

        if (i == this->idx){
            std::cerr << "Cannot add self egde"<<std::endl;
            throw std::runtime_error("Self edges not permitted");
        }
        int n = loglambdas.size();
        // n is the index of the new neighbor
        neighs.emplace(i, n);

        loglambdas.push_back(ArrayReal(0., this->T));

    }
    bool Node::set_contact(const int& j, const real_t& loglam, const uint& t){

        auto itr = neighs.find(j);
        if (itr == neighs.end())
            return false;
        const int& j_idx = itr->second;

        ArrayReal& llam = loglambdas.at(j_idx);
        if (t>=llam.size()) return false;
        llam[t] = loglam;

        return true;
    }
    bool Node::add_contact(const int& j,  const real_t& loglam, const uint& t){
        if (t>=this->T)
            throw std::range_error("t too big");
        if (!this->has_neigh(j)){
            this->add_neighbor(j);
        }
        return set_contact(j, loglam, t);
    }

    bool Node::has_neigh(const int& j){
        auto itr = neighs.find(j);
        if (itr == neighs.end())
            return false;
        else return true;
    }

}

#endif
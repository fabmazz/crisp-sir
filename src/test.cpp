#include <iostream>
#include <map>
#include "types.h"

int main ()
{
  auto node = Epi::Node(2);

  node.add_neighbor(5, 10);
  node.add_neighbor(3, 10);
  bool v = node.set_contact(8, -0.9, 3);
  if (!v){
    std::cout<<"ERROR: Cannot add contact"<<std::endl;
  }
  v = node.set_contact(5, -0.9, 9);
  if (!v){
    std::cout<<"ERROR: Cannot add contact"<<std::endl;
  }

  std::cout << "node contains:"<<std::endl;
  for (auto iter = node.neighs.begin(); iter!=node.neighs.end();++iter ){
      int idx = iter->second;

      std::cout << "neigh "<<iter->first<<" ";
      std::cout <<"\t times ";
      auto ll = node.loglambdas[idx];
      for (auto& x: ll)
        std::cout <<x<<" ";
    std::cout << std::endl;
  }

    
  std::cout << std::endl;

  return 0;
}
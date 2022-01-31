
#include <pybind11/pybind11.h>

#include "types.h"
namespace py = pybind11;

PYBIND11_MODULE(crisp, m){

    py::class_<Epi::Node>(m, "Node")
        .def(py::init<const int &, const uint&>())
        .def("add_neighbor", &Epi::Node::add_neighbor)
        .def("has_neighbor", &Epi::Node::has_neigh)
        .def("add_contact", &Epi::Node::add_contact,
        "add new contact to the node", py::arg("j"), py::arg("lambda"), py::arg("t"));
}
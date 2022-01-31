
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.h"
#include "crisp.cpp"
namespace py = pybind11;
using namespace std;
using namespace Epi;

PYBIND11_MAKE_OPAQUE(std::valarray<real_t>);
PYBIND11_MAKE_OPAQUE(std::vector<Node>);


vector<real_t> make_vector(py::list & l)
{
    vector<real_t> v(l.size());
    int i = 0;
    for (py::handle o : l) {
        v[i++] = py::cast<real_t>(o);
    }
    return v;
}

PYBIND11_MODULE(crisp, m){

        py::class_<ArrayReal>(m, "ArrayReal", py::buffer_protocol())
        .def(py::init([](py::buffer const b) {
                py::buffer_info info = b.request();
                if (info.format != py::format_descriptor<real_t>::format() || info.ndim != 1)
                throw std::runtime_error("Incompatible buffer format!");

                auto v = new ArrayReal(info.shape[0]);
                memcpy(&(*v)[0], info.ptr, sizeof(real_t) * (size_t) (v->size()));
                return v;
                }))
        .def(py::init([](vector<real_t> const & p)->ArrayReal {return ArrayReal(&p[0], p.size());}))
        .def(py::init([](py::list & l)->ArrayReal {auto v = make_vector(l); return ArrayReal(&v[0], v.size());}))
        .def_buffer([](ArrayReal &p) -> py::buffer_info {
            return py::buffer_info(
                &p[0],                               /* Pointer to buffer */
                sizeof(real_t),                          /* Size of one scalar */
                py::format_descriptor<real_t>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { p.size() },                 /* Buffer dimensions */
                { sizeof(real_t) }             /* Strides (in bytes) for each index */
                );
        })
        .def("__add__", [](ArrayReal & p, ArrayReal & q)->ArrayReal { return p + q; })
        .def("__getitem__", [](const ArrayReal &p, ssize_t i) {
                if (i > int(p.size()))
                    throw py::index_error();
                return p[i];
                })
        .def("__setitem__", [](ArrayReal &p, ssize_t i, real_t v) {
                if (i > int(p.size()))
                    throw py::index_error();
                p[i] = v;
                })
        .def("__repr__", [](ArrayReal &p) {
                    string s = "ArrayReal([";
                    for (size_t i = 0; i < p.size(); ++i)
                        s += (i ? ",":"") + tostring(p[i]);
                    s+="])";
                    return s;
                });

    py::class_<Epi::Node>(m, "Node")
        .def(py::init<const int &, const uint&>())
        .def("add_neighbor", &Epi::Node::add_neighbor)
        .def("has_neighbor", &Epi::Node::has_neigh)
        .def("add_contact", &Epi::Node::add_contact,
        "add new contact to the node", py::arg("j"), py::arg("lambda"), py::arg("t"));

    m.def("geometric_logp", &Epi::geometric_logp, "Give the geometric distribution (1-p)^(t-1) *p",
        py::arg("p"), py::arg("T"));
}
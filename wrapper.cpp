#include "graph_filter.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(graph_filter, m)
{
    m.doc() = "graph processing on point cloud";

    m.def("compute_scores", &compute_scores,
          "compute score for each point based on neighbors"); 
}
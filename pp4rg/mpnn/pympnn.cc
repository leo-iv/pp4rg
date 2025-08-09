#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ANN.h"
#include "multiann.h"

#ifndef PI
#define PI 3.1415926535897932385
#endif

namespace py = pybind11;

class KDTree {
  private:
    MPNN::MultiANN<int> *kdTree;

  public:
    KDTree(int dimension, py::array_t<int> topology, py::array_t<float> scale) {
        py::buffer_info topology_info = topology.request();
        py::buffer_info scale_info = scale.request();

        int *topology_arr = static_cast<int *>(topology_info.ptr);
        float *scale_arr = static_cast<float *>(scale_info.ptr);

        kdTree = new MPNN::MultiANN<int>(dimension, 1, topology_arr, scale_arr);
    }

    ~KDTree() { delete kdTree; }

    void add_point(py::array_t<float> coords, int index) {
        py::buffer_info coords_info = coords.request();
        float *coords_arr = static_cast<float *>(coords_info.ptr);
        kdTree->AddPoint(coords_arr, index);
    }

    py::tuple nearest_neighbor(py::array_t<float> query) {
        py::buffer_info query_info = query.request();
        float *query_arr = static_cast<float *>(query_info.ptr);

        int idx_mpnn;
        double dist;
        int my_idx = kdTree->NearestNeighbor(query_arr, idx_mpnn, dist);
        return py::make_tuple(my_idx, dist);
    }
};

/**
 * Helper function for calculating distance between two points with the given topology.
 */
double dist(py::array_t<float> point1, py::array_t<float> point2, py::array_t<int> topology, py::array_t<float> scale) {
    py::buffer_info topology_info = topology.request();
    py::buffer_info scale_info = scale.request();
    py::buffer_info point1_info = point1.request();
    py::buffer_info point2_info = point2.request();
    int *t = static_cast<int *>(topology_info.ptr);
    float *s = static_cast<float *>(scale_info.ptr);
    float *x1 = static_cast<float *>(point1_info.ptr);
    float *x2 = static_cast<float *>(point2_info.ptr);

    py::ssize_t dim = topology.size();

    double d = 0.0;
    for (int i = 0; i < dim; i++) {
        if (t[i] == 1) {
            d += ANN_POW(s[i] * (x1[i] - x2[i]));
        } else if (t[i] == 2) {
            double t = fabs(x1[i] - x2[i]);
            double t1 = ANN_MIN(t, 2.0 * PI - t);
            d += ANN_POW(s[i] * t1);
        } else if (t[i] == 3) {
            double fd = x1[i] * x2[i] + x1[i + 1] * x2[i + 1] + x1[i + 2] * x2[i + 2] + x1[i + 3] * x2[i + 3];
            if (fd > 1) {
                double norm1 = x1[i] * x1[i] + x1[i + 1] * x1[i + 1] + x1[i + 2] * x1[i + 2] + x1[i + 3] * x1[i + 3];
                double norm2 = x2[i] * x2[i] + x2[i + 1] * x2[i + 1] + x2[i + 2] * x2[i + 2] + x2[i + 3] * x2[i + 3];
                fd = fd / (norm1 * norm2);
            }
            double dtheta = ANN_MIN(acos(fd), acos(-fd));
            d += ANN_POW(s[i] * dtheta);
            i = i + 3;
        }
    }
    d = sqrt(d);
    return d;
}

PYBIND11_MODULE(mpnn, m) {
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<int, py::array_t<int>, py::array_t<float>>())
        .def("add_point", &KDTree::add_point)
        .def("nearest_neighbor", &KDTree::nearest_neighbor);
    m.def("dist", &dist);
}

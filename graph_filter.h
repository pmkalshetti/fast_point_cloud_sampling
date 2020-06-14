#include <Eigen/Eigen>
#include <Open3D/Open3D.h>
#include <omp.h>

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm>

using namespace Eigen;

using kdree_t = open3d::geometry::KDTreeFlann;
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixX3dR = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

void compute_resolution(const MatrixXd &points, const kdree_t &kdtree, double &min_dist, double &max_dist)
{
    min_dist = std::numeric_limits<double>::max();
    max_dist = std::numeric_limits<double>::min();
    double curr_dist{0.0};

#pragma omp parallel for
    for (long i = 0; i < points.rows(); ++i)
    {
        const int k{2};
        std::vector<int> ids;
        std::vector<double> dists_squared;
        kdtree.SearchKNN<Vector3d>(points.row(i), k, ids, dists_squared);
        assert(ids[0] == i || dists_squared[1] > 0.00000001);

#pragma omp critical
        {
            curr_dist = std::sqrt(dists_squared[1]);
            if (curr_dist > max_dist)
                max_dist = curr_dist;
            if (curr_dist < min_dist)
                min_dist = curr_dist;
        }
    }
}

SparseMatrix<double> compute_adjacency_matrix(const MatrixXd &points, const kdree_t &kdtree, const double radius, const int max_neighbors)
{
    const int n_points = points.rows();

    struct SparseElem
    {
        int row;
        int col;
        double value;
    };

    // init triplets vector array for each point
    std::vector<std::vector<SparseElem>> adj_lists(n_points);
    for (auto &adj_list : adj_lists)
        adj_list.reserve(max_neighbors);

// loop over each point
#pragma omp parallel for
    for (int i = 0; i < n_points; ++i)
    {
        // find ids and distances of neighbors in radius
        std::vector<int> ids;
        std::vector<double> dists_squared;
        int n_found = kdtree.SearchRadius<Vector3d>(points.row(i), radius, ids, dists_squared);
        if (!n_found)
            continue;

        // matched ids should be greater than id of current point
        assert(i == ids[0]);
#pragma omp critical
        for (int j = 1; j < n_found; ++j)
        {
            assert(dists_squared[j] > 0.00000001);

            // ensure id of point 1 < id of point 2
            int id_r;
            int id_c;
            if (i < ids[j])
            {
                id_r = i;
                id_c = ids[j];
            }
            else
            {
                id_r = ids[j];
                id_c = i;
            }

            // check if this pair is already filled
            bool pair_visited = false;
            for (int k = 0; k < adj_lists[id_r].size() && !pair_visited; ++k)
            {
                if (adj_lists[id_r][k].row == id_r && adj_lists[id_r][k].col == id_c)
                    pair_visited = true;
            }

            // create triplet and append to ith row
            if (!pair_visited && adj_lists[id_r].size() < max_neighbors)
            {
                adj_lists[id_r].push_back({id_r, id_c, dists_squared[j]});
            }

            // if ith row exceeds max size, then replace with lowest weight
            else if (!pair_visited && adj_lists[id_r].size() == max_neighbors)
            {
                // find farthest negihbor of current point in existing set
                double max_dist{0.0};
                int max_id{-1};
                for (int k = 0; k < max_neighbors; ++k)
                {
                    if (adj_lists[id_r][k].value > max_dist)
                    {
                        max_dist = adj_lists[id_r][k].value;
                        max_id = k;
                    }
                }

                // replace with new if closer than farthest
                if (max_dist > dists_squared[j])
                {
                    adj_lists[id_r][max_id].row = id_r;
                    adj_lists[id_r][max_id].col = id_c;
                    adj_lists[id_r][max_id].value = dists_squared[j];
                }
            }
        }
    }

    // compute variance
    double total_dist_squared{0.0};
    int n_entries{0};
    for (const auto &adj_list : adj_lists)
    {
        for (const auto &elem : adj_list)
            total_dist_squared += elem.value;
        n_entries += adj_list.size();
    }
    double avg_dist_squared{total_dist_squared / n_entries};

    double var_dist{0.0};
    for (const auto &adj_list : adj_lists)
    {
        for (const auto &elem : adj_list)
        {
            var_dist += std::pow(elem.value - avg_dist_squared, 2);
        }
    }
    var_dist /= n_entries;

    // construct W using above sparse elements (uses exp and variance)
    std::vector<Triplet<double>> triplets;
    triplets.reserve(n_entries);
    for (const auto &adj_list : adj_lists)
    {
        for (const auto &elem : adj_list)
        {
            triplets.push_back(Triplet<double>(elem.row, elem.col, std::exp(-std::pow(elem.value, 2) / (2 * var_dist))));
        }
    }
    SparseMatrix<double> W(n_points, n_points);
    W.setFromTriplets(triplets.begin(), triplets.end());
    W.makeCompressed();

    // fill the other triangle
    SparseMatrix<double> Wt = SparseMatrix<double>(W.transpose());
    W += Wt;

    return W;
}

VectorXd apply_filter(const MatrixXd &points, const SparseMatrix<double> &L)
{
    MatrixXd T = L * points;
    VectorXd scores = T.rowwise().norm();
    return scores;
}

SparseMatrix<double> compute_D(const SparseMatrix<double> &W)
{
    const int n_points = W.rows();  // or cols
    SparseMatrix<double> D(n_points, n_points);
    D.reserve(n_points); // diagonal
    for (int i = 0; i < W.outerSize(); ++i)
    {
        double row_sum{0.0};
        for (SparseMatrix<double>::InnerIterator it(W, i); it; ++it)
        {
            row_sum += it.value();
        }
        D.coeffRef(i, i) = row_sum;
    }

    return D;
}

VectorXd compute_scores(const MatrixXd &points, const std::string filter_type, const int scale_min_dist, const int scale_max_dist)
// VectorXd compute_scores(const MatrixXd &points, const std::string filter_type)
{
    const int n_points = points.rows();

    kdree_t kdtree(points.transpose()); // requires rows = dimension

    double min_dist{0.0};
    double max_dist{0.0};
    compute_resolution(points, kdtree, min_dist, max_dist);

    // const double radius = std::min(min_dist * 10, max_dist * 2);
    const double radius = std::min(min_dist * scale_min_dist, max_dist * scale_max_dist);
    std::cout << "\nResolution: " << min_dist << ", " << max_dist << ", " << radius << "\n";
    // const double radius = std::min(min_dist, max_dist);
    const int max_neighbors = 100;

    // adjacency matrix W
    SparseMatrix<double> W = compute_adjacency_matrix(points, kdtree, radius, max_neighbors);

    // // compute D
    // SparseMatrix<double> D(n_points, n_points);
    // D.reserve(n_points); // diagonal
    // for (int i = 0; i < W.outerSize(); ++i)
    // {
    //     double row_sum{0.0};
    //     for (SparseMatrix<double>::InnerIterator it(W, i); it; ++it)
    //     {
    //         row_sum += it.value();
    //     }
    //     D.coeffRef(i, i) = row_sum;
    // }

    // // compute L
    // SparseMatrix<double> L(n_points, n_points);
    // L = D - W;

    SparseMatrix<double> F(n_points, n_points);
    if (filter_type == "all")
    {
        F.setIdentity();
    }
    else if (filter_type == "high")
    {
        // F = D - W
        SparseMatrix<double> D = compute_D(W);
        F = D - W;
    }
    else if (filter_type == "low")
    {
        // F = D^-1 * W
        F = W;
        SparseMatrix<double> D = compute_D(W);
        for (int i = 0; i < F.outerSize(); ++i)
        {
            double row_sum{0.0};
            for (SparseMatrix<double>::InnerIterator it(F, i); it; ++it)
            {
                it.valueRef() *= D.coeff(i, i) + 1; 
            }
        }
    }
    else
    {
        std::cout << "ERROR! filter type has to be among {high, low, all}\n";
        std::exit(-1);   
    }

    // apply filter
    VectorXd scores = apply_filter(points, F);

    return scores;
}

std::vector<Vector3d> sample_points(const int n_samples, const std::vector<Vector3d> &points_vec, const std::string filter_type, const int scale_min_dist, const int scale_max_dist)
{
    const Map<MatrixXdR> points(const_cast<double *>(&points_vec[0](0)), points_vec.size(), 3);
    const VectorXd scores = compute_scores(points, filter_type, scale_min_dist, scale_max_dist);
    std::vector<double> scores_vec(scores.data(), scores.data()+scores.size());  // efficient to delete element in list
    std::random_device rd;
    std::mt19937 random_generator(rd());

    // double sum_score = std::accumulate(scores_vec.begin(), scores_vec.end(), 0.0);
    // std::for_each(scores_vec.begin(), scores_vec.end(), [sum_score](double &s){s /= sum_score;});
    // for (const auto& score : scores_vec)
    // {
    //     std::cout << score << "\n";
    // }
    // const double max_elem = *std::max_element(scores_vec.begin(), scores_vec.end());
    // std::cout << "max value = " << max_elem << "\n";
    // std::cout << "sum scores = " << std::accumulate(scores_vec.begin(), scores_vec.end(), 0.0) << "\n";
    // std::exit(0);

    // sample
    std::vector<Vector3d> sampled_points(n_samples);
    for (int i{0}; i < n_samples; ++i)
    {
        std::discrete_distribution prob(scores_vec.begin(), scores_vec.end());    // pi in paper
        int sampled_id = prob(random_generator);
        // if (scores_vec[sampled_id] < 0.0000001) continue;
        
        // std::cout << scores_vec[sampled_id] << ", ";
        scores_vec[sampled_id] = 0.0; // dont sample again
        // std::cout << scores_vec[sampled_id] << " ";
        // std::cout << std::accumulate(scores_vec.begin(), scores_vec.end(), 0.0) << "\n";

        sampled_points[i] = points.row(sampled_id);

    }

    return sampled_points;
}


std::vector<int> sample_ids(const int n_samples, const std::vector<Vector3d> &points_vec, const std::string filter_type, const int scale_min_dist, const int scale_max_dist)
{
    const Map<MatrixXdR> points(const_cast<double *>(&points_vec[0](0)), points_vec.size(), 3);
    VectorXd scores = compute_scores(points, filter_type, scale_min_dist, scale_max_dist);
    std::vector<double> scores_vec(scores.data(), scores.data()+scores.size());  // efficient to delete element in list
    std::random_device rd;
    std::mt19937 random_generator(rd());

    // sample
    std::vector<int> sampled_ids(n_samples);
    for (int i{0}; i < n_samples; ++i)
    {
        std::discrete_distribution prob(scores_vec.begin(), scores_vec.end());    // pi in paper
        int sampled_id = prob(random_generator);
        scores_vec[sampled_id] = 0.0; // dont sample again

        sampled_ids[i] = sampled_id;
    }

    return sampled_ids;
}
#ifndef LINALG_INCLUDE_EXPERIMENTAL___LAPACK_MANUAL_IMP_
#define LINALG_INCLUDE_EXPERIMENTAL___LAPACK_MANUAL_IMP_

namespace lapackman {

// 29.1 Cholesky factorization
// Returns nullopt if no bad pivots,
// else the index of the first bad pivot.
// A "bad" pivot is zero or NaN.
// lapack: xPOTRF2 (A=U'*U or L*L')
template <class InOutMat,
          class Triangle>
std::optional<typename InOutMat::size_type>
potrf2_cholesky_factor(InOutMat A, Triangle t)
{
    using value_type = typename InOutMat::value_type;
    using size_type = typename InOutMat::size_type;

    namespace la = std::experimental::linalg;
    namespace md = std;

    constexpr value_type ZERO{};
    constexpr value_type ONE(1.0);
    const size_type n = A.extent(0);

    if (n == 0)
    {
        return std::nullopt;
    }
    else if (n == 1)
    {
        if (A[0, 0] <= ZERO || std::isnan(A[0, 0]))
        {
            return {size_type(1)};
        }
        A[0, 0] = std::sqrt(A[0, 0]);
    }
    else
    {
        // Partition A into [A11, A12,
        //                   A21, A22],
        // where A21 is the transpose of A12.
        const size_type n1 = n / 2;
        const size_type n2 = n - n1;
        auto A11 = md::submdspan(A, std::pair{0, n1}, std::pair{0, n1});
        auto A22 = md::submdspan(A, std::pair{n1, n}, std::pair{n1, n});

        // Factor A11
        const auto info1 = potrf2_cholesky_factor(A11, t);
        if (info1.has_value())
        {
            return info1;
        }

        if constexpr (std::is_same_v<Triangle, la::upper_triangle_t>)
        {
            // Update and scale A12
            auto A12 = md::submdspan(A, std::tuple{0, n1}, std::tuple{n1, n});
            // BLAS would use original triangle; we need to flip it
            la::triangular_matrix_matrix_left_solve(la::transposed(A11), la::lower_triangle, la::explicit_diagonal, A12, A12);
            // A22 = A22 - A12^T * A12
            //
            // The Triangle argument applies to A22,
            // not transposed(A12), so we don't flip it.
            la::symmetric_matrix_rank_k_update(-ONE, la::transposed(A12), A22, t);
        }
        else
        {
            //
            // Compute the Cholesky factorization A = L * L^T
            //
            // Update and scale A21
            auto A21 = md::submdspan(A, std::tuple{n1, n}, std::tuple{0, n1});
            // BLAS would use original triangle; we need to flip it
            la::triangular_matrix_matrix_right_solve(la::transposed(A11), la::upper_triangle, la::explicit_diagonal, A21, A21);
            // A22 = A22 - A21 * A21^T
            la::symmetric_matrix_rank_k_update(-ONE, A21, A22, t);
        }

        // Factor A22
        const auto info2 = potrf2_cholesky_factor(A22, t);
        if (info2.has_value())
        {
            return {info2.value() + n1};
        }
    }

    return std::nullopt;
}

// 29.2 Solve linear system using Cholesky factorization
template <class InMat,
          class Triangle,
          class InVec,
          class OutVec>
void potrs_cholesky_solve(
    InMat A,
    Triangle t,
    InVec b,
    OutVec x)
{
    namespace la=std::experimental::linalg;

    if constexpr (std::is_same_v<Triangle, la::upper_triangle_t>)
    {
        // Solve Ax=b where A = U^T U
        //
        // Solve U^T c = b, using x to store c.
        la::triangular_matrix_vector_solve(la::transposed(A), la::lower_triangle, la::explicit_diagonal, b, x);
        // Solve U x = c, overwriting x with result.
        la::triangular_matrix_vector_solve(A, t, la::explicit_diagonal, x, x);
    }
    else
    {
        // Solve Ax=b where A = L L^T
        //
        // Solve L c = b, using x to store c.
        la::triangular_matrix_vector_solve(A, t, la::explicit_diagonal, b, x);
        // Solve L^T x = c, overwriting x with result.
        la::triangular_matrix_vector_solve(la::transposed(A), la::upper_triangle, la::explicit_diagonal, x, x);
    }
}

// 29.3 Compute QR factorization of a tall skinny matrix
//  Compute QR factorization A = Q R, with A storing Q.
//   This is just an example which has no real mathematical value.
//   Q/R is never computed in the way performed here.
template <class InOutMat,
          class OutMat>
std::optional<typename InOutMat::size_type>
cholesky_tsqr_one_step(
    InOutMat A, // A on input, Q on output
    OutMat R)
{
    using size_type = typename InOutMat::size_type;
    using index_type= typename InOutMat::index_type;
    using value_type= typename InOutMat::value_type;
    using value_type= typename InOutMat::value_type;
    using accessor_type= typename InOutMat::accessor_type;
    using R_value_type = typename OutMat::value_type;
    

    namespace la=std::experimental::linalg;
    namespace md=std;


    // One might use cache size, sizeof(element_type), and A.extent(1)
    // to pick the number of rows per block.  For now, we just pick
    // some constant.
    constexpr size_type max_num_rows_per_block = 500;
    constexpr R_value_type ZERO{};
    for (size_type j = 0; j < R.extent(1); ++j)
    {
        for (size_type i = 0; i < R.extent(0); ++i)
        {
            R[i, j] = ZERO;
        }
    }

    // Cache-blocked version of R = R + A^T * A.
    const auto num_rows = A.extent(0);
    auto rest_num_rows = num_rows;
    // auto A_rest = A;
    std::mdspan<value_type,
                std::dextents<index_type, 2>,
                std::layout_stride,
                accessor_type>
        A_rest=A;

    while (A_rest.extent(0) > 0)
    {
        const size_type num_rows_per_block = std::min(A.extent(0), max_num_rows_per_block);
        auto A_cur = md::submdspan(A_rest, std::tuple{0, num_rows_per_block}, md::full_extent);
        A_rest = md::submdspan(A_rest, std::tuple{num_rows_per_block, A_rest.extent(0)}, md::full_extent);
        // R = R + A_cur^T * A_cur
        constexpr R_value_type ONE(1.0);
        // The Triangle argument applies to R,
        // not transposed(A_cur), so we don't flip it.
        la::symmetric_matrix_rank_k_update(ONE, la::transposed(A_cur), R, la::upper_triangle);
    }

    const auto info = potrf2_cholesky_factor(R, la::upper_triangle);
    if (info.has_value())
    {
        return info;
    }

    la::triangular_matrix_matrix_right_solve(R, la::upper_triangle, la::explicit_diagonal, A, A);
    return std::nullopt;
}

// Compute QR factorization A = Q R.
// Use R_tmp as temporary R factor storage
// for iterative refinement.
//   This is just an example which has no real mathematical value.
//   Q/R is never computed in the way performed here.
template <class InMat,
          class OutMat1,
          class OutMat2,
          class OutMat3>
std::optional<typename OutMat1::size_type>
cholesky_tsqr(
    InMat A,
    OutMat1 Q,
    OutMat2 R_tmp,
    OutMat3 R)
{
    assert(R.extent(0) == R.extent(1));
    assert(A.extent(1) == R.extent(0));
    assert(R_tmp.extent(0) == R_tmp.extent(1));
    assert(A.extent(0) == Q.extent(0));
    assert(A.extent(1) == Q.extent(1));

    namespace la=std::experimental::linalg;

    la::copy(A, Q);
    const auto info1 = cholesky_tsqr_one_step(Q, R);
    if (info1.has_value())
    {
        return info1;
    }
    // Use one step of iterative refinement to improve accuracy.
    const auto info2 = cholesky_tsqr_one_step(Q, R_tmp);
    if (info2.has_value())
    {
        return info2;
    }
    // R = R_tmp * R
    la::triangular_matrix_left_product(R_tmp, la::upper_triangle, la::explicit_diagonal, R);
    return std::nullopt;
}

}

#endif


Note: This is just a fork of [Kokos/StdBlas](https://github.com/kokkos/stdBLAS).

The original code uses paranthesis operator for indexing. 
I only modified a couple of lines so that the library can use multi-index operator based on the value of MDSPAN_USE_BRACKET_OPERATOR.
#include "common.cuh"
#include "maxpool.cuh"
#include "test_utils.cuh"

#include <iostream>
#include <vector>

using namespace nnv2;

void test_maxpool_forward() {
    int batch_size = 2;
    int in_feats = 2;
    int in_h = 4;
    int in_w = 4;

    // test max pooling with no padding
    Array input({2, 2, 4, 4}, {1, 3, 2, 1, 4, 6, 5, 1, 1, 2, 1, 3, 0, 2, 4, 1,
                               0, 1, 1, 0, 1, 0, 2, 1, 2, 3, 1, 2, 1, 0, 1, 3,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               5, 4, 3, 2, 4, 5, 6, 1, 3, 6, 5, 4, 2, 1, 4, 3});

    Array output({2, 2, 2, 2});
    Array indices({2, 2, 2, 2});

    maxpool_forward(&output, &input, &indices, 0, 0, 2, 2, 2, 2);
    check_equal_vecs(output.get_vec(),
                     {6, 5, 2, 4, 1, 2, 3, 3, 2, 2, 2, 2, 5, 6, 6, 5});

    // main test
    int pad_h = 1;
    int pad_w = 1;
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    output.resize({batch_size, in_feats, out_h, out_w});
    indices.resize({batch_size, in_feats, out_h, out_w});

    maxpool_forward(&output, &input, &indices, pad_h, pad_w, kernel_h, kernel_w,
                    stride_h, stride_w);
    check_equal_vecs(output.get_vec(),
                     {1, 3, 1, 4, 6, 3, 0, 4, 1, 0, 1, 0, 2, 3, 2, 1, 1, 3,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 4, 2, 4, 6, 4, 2, 4, 3});

    std::cout << "test_maxpool_forward: Passed" << std::endl;
}

void test_maxpool_forward_case2() {
    int batch_size = 2;
    int in_feats = 2;
    int in_h = 3;
    int in_w = 3;

    // Should pad a row to the bottom and a column to the right
    //
    // 1 3 2 .   0 1 1 .   2 2 2 .   5 4 3 .
    // 1 4 6 .   0 1 0 .   2 2 2 .   2 4 5 .
    // 5 1 1 .   2 1 2 .   2 2 2 .   6 1 3 .
    // . . . .   . . . .   . . . .   . . . .
    //
    // After pooling, it should be
    // 4 6   1 1   2 2   5 5
    // 5 1   2 2   2 2   6 3
    Array input({2, 2, 3, 3}, {1, 3, 2, 1, 4, 6, 5, 1, 1,
                               0, 1, 1, 0, 1, 0, 2, 1, 2,
                               2, 2, 2, 2, 2, 2, 2, 2, 2,
                               5, 4, 3, 2, 4, 5, 6, 1, 3});

    Array output({2, 2, 2, 2});
    Array indices({2, 2, 2, 2});

    maxpool_forward(&output, &input, &indices, 1, 1, 2, 2, 2, 2);
    check_equal_vecs(output.get_vec(),
                     {4, 6, 5, 1, 1, 1, 2, 2, 2, 2, 2, 2, 5, 5, 6, 3});

    std::cout << "test_maxpool_forward_case2: Passed" << std::endl;
}

void test_maxpool_backward() {
    Array input({2, 2, 4, 4}, {1, 3, 2, 1, 4, 6, 5, 1, 1, 2, 1, 3, 0, 2, 4, 1,
                               0, 1, 1, 0, 1, 0, 2, 1, 2, 3, 1, 2, 1, 0, 1, 3,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               5, 4, 3, 2, 4, 5, 6, 1, 3, 6, 5, 4, 2, 1, 4, 3});

    Array output({2, 2, 3, 3});
    Array indices({2, 2, 3, 3});

    Array input_grad({2, 2, 4, 4});
    Array output_grad({2, 2, 3, 3},
                      {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    maxpool_forward(&output, &input, &indices, 1, 1, 2, 2, 2, 2);
    maxpool_backward(&input_grad, &output_grad, &indices, 1, 1, 2, 2, 2, 2);
    check_equal_vecs(input_grad.get_vec(),
                     {1, 2, 0, 3, 4, 5, 0, 0, 0, 0, 0, 6, 7, 0, 8, 9,
                      1, 2, 0, 3, 0, 0, 0, 0, 4, 5, 0, 6, 7, 0, 8, 9,
                      1, 2, 0, 3, 4, 5, 0, 6, 0, 0, 0, 0, 7, 8, 0, 9,
                      1, 2, 0, 3, 4, 0, 5, 0, 0, 0, 0, 6, 7, 0, 8, 9});
    std::cout << "test_maxpool_back: Passed" << std::endl;
}

int main() {
    test_maxpool_forward();
    test_maxpool_forward_case2();
    test_maxpool_backward();
}
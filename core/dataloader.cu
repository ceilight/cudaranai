// This file implements the data loader
// The data loader acts as the first layer in the neural network, handling the
// distribution of dataset during training and testing phases, with tasks such
// as loading batch data and shuffle the dataset

#include "common.cuh"
#include "dataloader.cuh"

#include <algorithm>
#include <iostream>
#include <vector>

#include <thrust/copy.h>

namespace nnv2 {

int DataLoader::load_train_batch() {
    // update offset
    int start = train_data_offset;
    int end =
        std::min(start + batch_size, (int)dataset->get_train_images().size());
    train_data_offset = end;
    int size = end - start;

    // initialize batch memory
    int h = dataset->get_image_height();
    int w = dataset->get_image_width();
    int n_labels = dataset->get_label_count();
    set_array_ptr(output, {size, 1, h, w});
    set_array_ptr(output_labels, {size, n_labels});

    // extract a batch of train data
    int im_stride = h * w;
    for (int i = start; i < end; i++) {
        // train images
        thrust::copy(dataset->get_train_images()[i].begin(),
                     dataset->get_train_images()[i].end(),
                     output->get_vec().begin() + (i - start) * im_stride);

        // train labels, with one-hot encoding
        int one_hot_index =
            (i - start) * n_labels + (int)dataset->get_train_labels()[i];
        output_labels->get_vec()[one_hot_index] = 1;
    }

    return size;
}

int DataLoader::load_test_batch() {
    // update offset
    int start = test_data_offset;
    int end =
        std::min(start + batch_size, (int)dataset->get_test_images().size());
    test_data_offset = end;
    int size = end - start;

    // initialize batch memory
    int h = dataset->get_image_height();
    int w = dataset->get_image_width();
    int n_labels = dataset->get_label_count();
    set_array_ptr(output, {size, 1, h, w});
    set_array_ptr(output_labels, {size, n_labels});

    // extract a batch of test data
    int im_stride = h * w;
    for (int i = start; i < end; i++) {
        // test images
        thrust::copy(dataset->get_test_images()[i].begin(),
                     dataset->get_test_images()[i].end(),
                     output->get_vec().begin() + (i - start) * im_stride);

        // test labels, with one-hot encoding
        int one_hot_index =
            (i - start) * n_labels + (int)dataset->get_test_labels()[i];
        output_labels->get_vec()[one_hot_index] = 1;
    }

    return size;
}

bool DataLoader::has_next_train_batch() {
    return train_data_offset < dataset->get_train_images().size();
}

bool DataLoader::has_next_test_batch() {
    return test_data_offset < dataset->get_test_images().size();
}

void DataLoader::reset(bool shuffle) {
    train_data_offset = 0;
    test_data_offset = 0;
    if (shuffle) {
        dataset->shuffle_train_data();
    }
}

} // namespace nnv2
//
// Created by fss on 23-5-27.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/tensor.hpp"

int main(int argc, char *argv[]) {
    // my_infer::Tensor<float>(10);

    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("Kuiper");
    FLAGS_log_dir = "../../course2/log";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}

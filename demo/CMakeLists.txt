add_executable(EXTRACTION ../extraction/au_file_processor.cpp extractor_demo.cpp)
add_executable(CART ../ml_algorithms/decision_tree.cpp decision_tree_demo.cpp)
add_executable(RANDOM_FOREST ../ml_algorithms/decision_tree.cpp ../ml_algorithms/random_forest.cpp random_forest_demo.cpp)
add_executable(SVM ../ml_algorithms/one_vs_one_svm.cpp one_vs_one_svm_demo.cpp)
add_executable(ANN ../ml_algorithms/artificial_neural_network.cpp artificial_neural_network_demo.cpp)

# Link against the dependency of Intel TBB (for parallel C++ algorithms)
# target_link_libraries(PROJECT tbb)
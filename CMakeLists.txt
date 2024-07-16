cmake_minimum_required(VERSION 3.16)
project(my_infer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)
find_package(benchmark REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)

set(link_lib glog::glog GTest::gtest)
set(link_math_lib ${ARMADILLO_LIBRARiES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# 将test和source目录下的所有文件都保存在两个变量中，add_excutable时就可以直接引用这两个变量，
# 而不需要一一列举test和source目录下所有的文件
aux_source_directory(./test DIR_TEST_ARMA)
# aux_source_directory(./source DIR_SOURCE_ARMA)
aux_source_directory(./source/data DIR_DATA)
# 将上面收集到的test和source文件以及main.cpp文件一起编译为my_infer可执行文件
# 这里只需要包含源文件，不需要引用头文件，因为它们通过include被源文件引用，编译器会自动查找头文件
add_executable(my_infer main.cpp ${DIR_TEST_ARMA} ${DIR_DATA})
target_link_libraries(my_infer ${link_lib} ${link_math_lib})

# 用来指定搜索的头文件的路径，include的时候就不需要将库文件与当前文件的相对位置完完整整写清楚,
# 直接写include的文件名即可，然后编译器会从target_include_directories指定的目录下开始搜索
target_include_directories(my_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(my_infer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(my_infer PUBLIC ${Armadillo_INCLUDE_DIR})
# 上面的target_include_directories仅仅对编译器有效，但是写代码时clang依然无法识别头文件的路径
# 所以需要加上下面的代码，会生成compile_commands.json文件，现在include就不会爆红了
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

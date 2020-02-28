add_library(reconstructor STATIC
  reconstructor/Camera.h
  reconstructor/Camera.cpp
  reconstructor/ForegroundOptimizer.h
  reconstructor/ForegroundOptimizer.cpp
  reconstructor/Reconstructor.h
  reconstructor/Reconstructor.cpp
)
target_include_directories(reconstructor INTERFACE reconstructor/)
target_link_libraries(reconstructor PRIVATE ${OpenCV_LIBS} Threads::Threads OpenMP::OpenMP_CXX)

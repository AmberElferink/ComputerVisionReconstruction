add_executable(background_averager background_averager.cpp)
target_link_libraries(background_averager PRIVATE ${OpenCV_LIBS})

add_executable(background_averager background_averager.cpp)
target_link_libraries(background_averager PRIVATE ${OpenCV_LIBS})

add_executable(camera_calibrator camera_calibrator.cpp)
target_link_libraries(camera_calibrator PRIVATE ${OpenCV_LIBS})

add_executable(image_extractor image_extractor.cpp)
target_link_libraries(image_extractor PRIVATE ${OpenCV_LIBS})

add_executable(hand_calibration checkerboard_marking.cpp)
target_link_libraries(hand_calibration PRIVATE ${OpenCV_LIBS})

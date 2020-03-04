cmake_minimum_required(VERSION 3.10)
project(VoxelRecontruction)

# set(CMAKE_VERBOSE_MAKEFILE OFF)
set (CMAKE_CXX_STANDARD 17)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# if(CMAKE_BUILD_TYPE MATCHES Debug)
#         add_definitions(-O0)
#         add_definitions(-g3)
#         add_definitions(-DDEBUG)
# else(CMAKE_BUILD_TYPE MATCHES Debug)
#         set(CMAKE_BUILD_TYPE Release)
#         add_definitions(-O3)
#         add_definitions(-g0)
#         add_definitions(-DNDEBUG)
# endif(CMAKE_BUILD_TYPE MATCHES Debug)
# 
# add_definitions(-DTIXML_USE_TICPP)

find_package(SDL2 CONFIG)
if (WIN32 AND MINGW)
	set(SDL2_INCLUDE_DIRS /mingw64/include/SDL2)
elseif (TARGET SDL2::SDL2)
	set(SDL2_LIBRARIES SDL2::SDL2)
endif()
find_package(glm REQUIRED)
find_package(OpenGL 1 REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc calib3d ml REQUIRED)
find_package(Threads)
find_package(OpenMP COMPONENTS CXX REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
# 
# set(OpenCV_USE_SHARED_LIBS ON)
# 
# #set the default path for built executables to the BUILD_TYPE directory ("Release" or "Debug")
# file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
# 
# #############################################

#states where other CMakeLists.txt are stored to make the dependent executables
add_subdirectory(libs)
add_subdirectory(apps)
add_subdirectory(thirdparty/glad)
add_subdirectory(thirdparty/imgui-1.75)

set(SOURCE_FILES
	src/graphics/Buffer.cpp
	src/graphics/Mesh.cpp
	src/graphics/Pipeline.cpp
	src/graphics/private_impl/ComputePipeline.cpp
	src/graphics/private_impl/GraphicsPipeline.cpp
	src/graphics/Context.cpp
	src/graphics/RenderPass.cpp
	src/graphics/Texture.cpp
	src/graphics/Ui.cpp
	src/controllers/ArcBall.cpp
	src/controllers/Renderer.cpp
	src/controllers/Scene3DRenderer.cpp
	src/main.cpp
	src/VoxelReconstruction.cpp)

#############################################
# Videos are at 50 FPS so step by 50 frames for one capture per second to save time
# as the videos are on the order of minutes long.
set(CALIBRATION_FRAME_STEP 50 CACHE STRING "How many subsequent frames to skip during calibration")
# Set each data file needed as a "source file" then tell cmake how those files
# are generated. For some, it is only a matter of copying. For others, it requires
# one of the apps to run to generate the dependency files. CMake will take care
# of the dependencies.
list(APPEND FILES_TO_COPY data/checkerboard.xml)
foreach(CAMERA cam1 cam2 cam3 cam4)
	list(APPEND FILES_TO_COPY data/${CAMERA}/video.avi)
	list(APPEND FILES_TO_COPY data/${CAMERA}/checkerboard.avi)
	list(APPEND VIDEOS data/${CAMERA}/video.avi)
	list(APPEND CHECKERBOARDS data/${CAMERA}/checkerboard.avi)
	list(APPEND BACKGROUNDS data/${CAMERA}/background.avi)
	list(APPEND INTRINSICS data/${CAMERA}/intrinsics.avi)
endforeach()
set(COLOR_CALIBRATION_FRAME_NUMBER 1180)

# Tell cmake to copy video files in order to include them
foreach(asset ${FILES_TO_COPY})
	get_filename_component(asset_name ${asset} NAME)
	get_filename_component(asset_directory ${asset} DIRECTORY)
	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/${asset_name})
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${asset} ${asset_output}
		DEPENDS ${asset}
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

# Tell cmake to generate background.png files
foreach(asset ${BACKGROUNDS})
	get_filename_component(asset_name_we ${asset} NAME_WE)
	get_filename_component(asset_directory ${asset} DIRECTORY)
	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/${asset_name_we}.png)
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND background_averager ${CMAKE_CURRENT_SOURCE_DIR}/${asset} ${asset_output}
		DEPENDS ${asset}
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

# Tell cmake to generate color_calibration.png files
foreach(asset ${VIDEOS})
	get_filename_component(asset_name_we ${asset} NAME_WE)
	get_filename_component(asset_directory ${asset} DIRECTORY)
	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/color_calibration.png)
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND image_extractor ${CMAKE_CURRENT_SOURCE_DIR}/${asset} ${asset_output} ${COLOR_CALIBRATION_FRAME_NUMBER}
		DEPENDS ${asset}
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

# Tell cmake to generate intrinsics.xml files
foreach(asset ${INTRINSICS})
	get_filename_component(asset_name_we ${asset} NAME_WE)
	get_filename_component(asset_directory ${asset} DIRECTORY)
	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/intrinsics.xml)
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND camera_calibrator ${CMAKE_CURRENT_SOURCE_DIR}/data/checkerboard.xml ${CMAKE_CURRENT_SOURCE_DIR}/${asset} ${asset_output} ${CALIBRATION_FRAME_STEP}
		DEPENDS ${asset} data/checkerboard.xml
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

# Tell cmake to generate boardcorners.xml by first averaging checkerboard.avi files
foreach(asset ${CHECKERBOARDS})
	get_filename_component(asset_directory ${asset} DIRECTORY)
	get_filename_component(asset_name_we ${asset} NAME_WE)

	# Average the checkboard video to image
	set(checkerboard_image ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/${asset_name_we}.png)
	add_custom_command(
		OUTPUT ${checkerboard_image}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND background_averager ${asset} ${checkerboard_image}
		DEPENDS ${asset}
	)

	# Have user calibrate by hand
	set(corners ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/boardcorners.xml)
	add_custom_command(
		OUTPUT ${corners}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND hand_calibration ${CMAKE_CURRENT_SOURCE_DIR}/data/checkerboard.xml ${checkerboard_image} ${corners}
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/checkerboard.xml ${checkerboard_image}
	)

	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/config.xml)
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND extrinsics_configurator ${CMAKE_CURRENT_SOURCE_DIR}/data/checkerboard.xml ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/intrinsics.xml ${checkerboard_image} ${corners} ${asset_output}
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/checkerboard.xml ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/intrinsics.xml ${checkerboard_image} ${corners}
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

#############################################
add_executable (${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

#############################################
target_link_libraries (${CMAKE_PROJECT_NAME}
	PRIVATE
		glad
		imgui
		glm
		reconstructor #this refers to a dependency on our own code (libs/CMakeLists.txt)
		OpenGL::GL
		${SDL2_LIBRARIES}
		${OpenCV_LIBS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})

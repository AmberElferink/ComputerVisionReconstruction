cmake_minimum_required(VERSION 3.10)
project(VoxelRecontruction)

# set(CMAKE_VERBOSE_MAKEFILE OFF)
set (CMAKE_CXX_STANDARD 11)

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

find_package(FreeGLUT REQUIRED)
find_package(OpenGL 1 REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc calib3d REQUIRED)
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

set(SOURCE_FILES
	src/controllers/arcball.cpp
	src/controllers/Camera.cpp
	src/controllers/Glut.cpp
	src/controllers/Reconstructor.cpp
	src/controllers/Scene3DRenderer.cpp
	src/main.cpp
	src/utilities/General.cpp
	src/VoxelReconstruction.cpp)

# Tell cmake to copy data files in order to include them
set(ASSETS
	data/cam1/video.avi
)
foreach(asset ${ASSETS})
	get_filename_component(asset_name ${asset} NAME)
	get_filename_component(asset_directory ${asset} DIRECTORY)
	message(${asset_name})
	set(asset_output ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}/${asset_name})
	add_custom_command(
		OUTPUT ${asset_output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${asset_directory}
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${asset} ${asset_output}
		DEPENDS ${asset}
	)
	list(APPEND SOURCE_FILES ${asset_output})
endforeach()

add_executable (${CMAKE_PROJECT_NAME} ${SOURCE_FILES} ${ASSETS})

#############################################
target_link_libraries (${CMAKE_PROJECT_NAME}
	PRIVATE
		OpenGL::GL
		OpenGL::GLU
		FreeGLUT::freeglut
		Threads::Threads
		OpenMP::OpenMP_CXX
		${OpenCV_LIBS})

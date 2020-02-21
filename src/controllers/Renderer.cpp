/*
 * Glut.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: Coert and a guy named Frank
 */

#include "Renderer.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <SDL2/SDL.h>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "ArcBall.h"
#include "Camera.h"
#include "Reconstructor.h"
#include "Scene3DRenderer.h"
#include "../graphics/Buffer.h"
#include "../graphics/Context.h"
#include "../graphics/Mesh.h"
#include "../graphics/Pipeline.h"
#include "../graphics/RenderPass.h"
#include "../graphics/Ui.h"
#include "../graphics/Texture.h"
#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

struct DrawArraysIndirectCommand {
  uint32_t count;
  uint32_t instanceCount;
  uint32_t first;
  uint32_t baseInstance;
};

constexpr std::string_view viewProjVertexShaderSource =
	"#version 450 core\n"
	"layout (location = 0) in vec3 position;\n"
	"uniform mat4 view;\n"
	"uniform mat4 proj;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = proj * view * vec4(position, 1.0);\n"
	"}\n";

constexpr std::string_view gridFragmentShaderSource =
	"#version 450 core\n"
	"layout (location = 0) out vec4 out_color;\n"
	"void main()\n"
	"{\n"
	"    out_color = vec4(0.9f, 0.9f, 0.9f, 0.5f);\n"
	"}\n";

constexpr std::string_view arcballVertexShaderSource =
	"#version 450 core\n"
	"layout (location = 0) in vec3 position;\n"
	"uniform float scale;\n"
	"uniform mat4 view;\n"
	"uniform mat4 proj;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = proj * view * vec4(scale * position, 1.0);\n"
	"}\n";

constexpr std::string_view arcballFragmentShaderSource =
	"#version 450 core\n"
	"layout (location = 0) out vec4 out_color;\n"
	"void main()\n"
	"{\n"
	"    out_color = vec4(1.0f, 0.9f, 0.9f, 1.0f);\n"
	"}\n";

constexpr std::string_view wVertexShaderSource =
	"#version 450 core\n"
	"layout (location = 0) in vec4 position;\n"
	"layout (location = 1) in vec4 color;\n"
	"layout (location = 0) out vec4 out_color;\n"
	"uniform mat4 view;\n"
	"uniform mat4 proj;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = proj * view * position;\n"
	"    out_color = color;\n"
	"}\n";

constexpr std::string_view wFragmentShaderSource =
	"#version 450 core\n"
	"layout (location = 0) in vec4 color;\n"
	"layout (location = 0) out vec4 out_color;\n"
	"void main()\n"
	"{\n"
	"    out_color = color;\n"
	"}\n";


// Table from http://paulbourke.net/geometry/polygonise
constexpr uint32_t marchingCubeEdgeTable[256]={
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
constexpr int marchingCubeTriTable[256][16] =
	{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

constexpr std::string_view marchingCubeComputeShaderSource =
	"#version 450 core\n"
	"layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;\n"
	"uniform vec3 resolution;\n"
	"uniform sampler3D scalar_field;\n"
	"struct DrawArraysIndirectCommand {\n"
	"    uint count;\n"
	"    uint instanceCount;\n"
	"    uint first;\n"
	"    uint baseInstance;\n"
	"};\n"
	"layout(std430, binding = 0) buffer indirect_data\n"
	"{\n"
	"    DrawArraysIndirectCommand indirectData;\n"
	"};\n"
	"layout(std430, binding = 1) buffer edge_lut\n"
	"{\n"
	"    uint edgeTable[256];\n"
	"};\n"
	"layout(std430, binding = 2) buffer triangle_lut\n"
	"{\n"
	"    int triTable[4096];\n"
	"};\n"
	"layout(std430, binding = 3) buffer vertex_data\n"
	"{\n"
	"    vec4 vertices[];\n"
	"};\n"
	"vec4 positions[8] = {\n"
	"    vec4(0, 0, 0, 1),\n"
	"    vec4(1, 0, 0, 1),\n"
	"    vec4(1, 0, 1, 1),\n"
	"    vec4(0, 0, 1, 1),\n"
	"    vec4(0, 1, 0, 1),\n"
	"    vec4(1, 1, 0, 1),\n"
	"    vec4(1, 1, 1, 1),\n"
	"    vec4(0, 1, 1, 1)};\n"
	"\n"
	"void main()\n"
	"{\n"
	"    // Reset indirect draw buffer only on shader core 0 and synchronize\n"
	"    if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {\n"
	"        indirectData.count = 0;\n"
	"        indirectData.instanceCount = 1;\n"
	"        indirectData.first = 0;\n"
	"        indirectData.baseInstance = 0;\n"
	"    }\n"
	"    memoryBarrierBuffer();\n"
	"    if (gl_GlobalInvocationID.x >= resolution.x - 1 || gl_GlobalInvocationID.y >= resolution.y - 1 || gl_GlobalInvocationID.z >= resolution.z - 1) {\n"
	"        return;\n"
	"    }\n"
	"    uint classification = 0;\n"
	"    for (uint i = 0; i < 8; ++i) {\n"
	"        uint bit = 1 << i;\n"
	"        if (texelFetch(scalar_field, ivec3(gl_GlobalInvocationID) + ivec3(positions[i].xyz), 0).r > 0.5f) {\n"
	"            classification |= bit;\n"
	"        }\n"
	"    }\n"
	"    uint edge_mask = edgeTable[classification];\n"
	"    vec4 vertlist[12] = {\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0),\n"
	"        vec4(0, 0, 0, 0)};\n"
	"    if (edge_mask != 0) {\n"
	"        /* Find the vertices where the surface intersects the cube */\n"
	"        if ((edgeTable[classification] & 1) != 0)\n"
	"           vertlist[0] = mix(positions[0], positions[1], 0.5f);\n"
	"        if ((edgeTable[classification] & 2) != 0)\n"
	"           vertlist[1] = mix(positions[1], positions[2], 0.5f);\n"
	"        if ((edgeTable[classification] & 4) != 0)\n"
	"           vertlist[2] = mix(positions[2], positions[3], 0.5f);\n"
	"        if ((edgeTable[classification] & 8) != 0)\n"
	"           vertlist[3] = mix(positions[3], positions[0], 0.5f);\n"
	"        if ((edgeTable[classification] & 16) != 0)\n"
	"           vertlist[4] = mix(positions[4], positions[5], 0.5f);\n"
	"        if ((edgeTable[classification] & 32) != 0)\n"
	"           vertlist[5] = mix(positions[5], positions[6], 0.5f);\n"
	"        if ((edgeTable[classification] & 64) != 0)\n"
	"           vertlist[6] = mix(positions[6], positions[7], 0.5f);\n"
	"        if ((edgeTable[classification] & 128) != 0)\n"
	"           vertlist[7] = mix(positions[7], positions[4], 0.5f);\n"
	"        if ((edgeTable[classification] & 256) != 0)\n"
	"           vertlist[8] = mix(positions[0], positions[4], 0.5f);\n"
	"        if ((edgeTable[classification] & 512) != 0)\n"
	"           vertlist[9] = mix(positions[1], positions[5], 0.5f);\n"
	"        if ((edgeTable[classification] & 1024) != 0)\n"
	"           vertlist[10] = mix(positions[2], positions[6], 0.5f);\n"
	"        if ((edgeTable[classification] & 2048) != 0)\n"
	"           vertlist[11] = mix(positions[3], positions[7], 0.5f);\n"
	"    }\n"
	"    for (uint i = 0; triTable[classification * 16 + i] != -1; i += 3) {\n"
	"        uint index = atomicAdd(indirectData.count, 3);\n"
	"        vertices[index] = vertlist[triTable[classification * 16 + i]] + vec4(gl_GlobalInvocationID, 0);\n"
	"        vertices[index + 1] = vertlist[triTable[classification * 16 + i + 1]] + vec4(gl_GlobalInvocationID, 0);\n"
	"        vertices[index + 2] = vertlist[triTable[classification * 16 + i + 2]] + vec4(gl_GlobalInvocationID, 0);\n"
	"    }\n"
	"}\n";

constexpr std::string_view voxelVertexShaderSource =
	"#version 450 core\n"
	"uniform mat4 view;\n"
	"uniform mat4 proj;\n"
	"uniform float scale;\n"
	"uniform vec3 offset;\n"
	"layout (location = 0) in vec4 position;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = proj * view * vec4(position.xyz * scale + offset, position.w);\n"
	"}\n";

constexpr std::string_view voxelFragmentShaderSource =
	"#version 450 core\n"
	"layout (location = 0) out vec4 out_color;\n"
	"void main()\n"
	"{\n"
	"    out_color = vec4(0.5f, 0.5f, 0.5f, 0.5f);\n"
	"}\n";

Renderer::Renderer(Scene3DRenderer &s3d)
	: m_scene3d(s3d)
	, m_arc_ball()
	, m_viewMatrix(1)
	, m_projectionMatrix(1)
{
}

Renderer::~Renderer() = default;

void Renderer::initialize(const char* win_name, int argc, char** argv)
{
	m_renderer = Context::create(win_name, 700, 10, m_scene3d.getWidth(), m_scene3d.getHeight());
	m_ui = Ui::create(m_renderer->getNativeWindowHandle());
	RenderPass::CreateInfo passInfo;
	passInfo.Clear = true;
	passInfo.ClearColor[0] = 1.0f;
	passInfo.ClearColor[1] = 1.0f;
	passInfo.ClearColor[2] = 1.0f;
	passInfo.ClearColor[3] = 1.0f;
	passInfo.DepthWrite = false;
	passInfo.DepthTest = false;
	passInfo.DebugName = "Main Render Pass";
	m_renderPass = RenderPass::create(passInfo);

	initializeGeometry();
	initializePipelines();
	m_projectionMatrix = glm::perspective(glm::radians(50.0f), m_renderer->getAspectRatio(), 1.0f, 40000.0f);

	reset(); //initialize the ArcBall for scene rotation

	bool running = true;
	SDL_Event event;
	bool mouse_down = false;

	while (running) {
		while (SDL_PollEvent(&event)) {
			m_ui->processEvent(event);
			switch (event.type) {
			case SDL_QUIT: // cross
				running = false;
				break;
			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_RESIZED)
				{
					int width, height;
					m_renderer->getSize(width, height);
					reshape(width, height);
				} break;
			case SDL_MOUSEWHEEL:
			case SDL_MOUSEBUTTONDOWN:
			case SDL_MOUSEBUTTONUP:
			case SDL_MOUSEMOTION:
				mouse(event, mouse_down);
				break;
			case SDL_KEYDOWN:
				keyboard(event.key);
				break;
			}
		}
		update();
		display();
	}
	quit();
}

void Renderer::initializeGeometry()
{
	Mesh::CreateInfo info = {};
	const std::vector<Mesh::MeshAttributes> wireframe_attributes{
		Mesh::MeshAttributes{Mesh::MeshAttributes::DataType::Float, 3},
	};

	// Grid
	{
		int gSize = m_scene3d.getNum() * 2 + 1;

		// Vertex Buffer
		Buffer::CreateInfo buffer_info;
		buffer_info.Size = 4 * gSize * sizeof(glm::vec3);
		buffer_info.BufferType = Buffer::Type::Vertex;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "grid vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);

		vector<vector<Point3i *> > floor_grid = m_scene3d.getFloorGrid();
		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.MeshTopology = Mesh::Topology::Lines;
		info.DebugName = "grid";
		m_gridMesh = Mesh::create(info, std::move(vertex_buffer), nullptr);

		auto vertexMem = m_gridMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		for (int g = 0; g < gSize; g++)
		{
			auto vertexBase = &reinterpret_cast<glm::vec3*>(vertexMem.get())[g * 4];
			vertexBase[0] = glm::vec3(floor_grid[0][g]->x, floor_grid[0][g]->y, floor_grid[0][g]->z);
			vertexBase[1] = glm::vec3(floor_grid[2][g]->x, floor_grid[2][g]->y, floor_grid[2][g]->z);
			vertexBase[2] = glm::vec3(floor_grid[1][g]->x, floor_grid[1][g]->y, floor_grid[1][g]->z);
			vertexBase[3] = glm::vec3(floor_grid[3][g]->x, floor_grid[3][g]->y, floor_grid[3][g]->z);
		}
	}

	// Camera
	{
		vector<Camera*> cameras = m_scene3d.getCameras();

		Buffer::CreateInfo buffer_info;
		// Vertex Buffer
		buffer_info.Size = 5 * cameras.size() * sizeof(glm::vec3);
		buffer_info.BufferType = Buffer::Type::Vertex;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "cameras vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);
		// Index Buffer
		buffer_info.Size = 16 * cameras.size() * sizeof(uint32_t);
		buffer_info.BufferType = Buffer::Type::Index;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "cameras index buffer";
		auto index_buffer = Buffer::create(buffer_info);

		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.MeshTopology = Mesh::Topology::Lines;
		info.DebugName = "cameras";
		m_cameraMesh = Mesh::create(info, std::move(vertex_buffer), std::move(index_buffer));
		auto vertexMem = m_cameraMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexMem = m_cameraMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);
		for (size_t i = 0; i < cameras.size(); i++)
		{
			vector<Point3f> plane = cameras[i]->getCameraPlane();
			// 16 indices * camera.size()
			auto indexBase = &reinterpret_cast<uint32_t*>(indexMem.get())[i * 16];
			indexBase[0] = i * 5;
			indexBase[1] = i * 5 + 1;
			indexBase[2] = i * 5;
			indexBase[3] = i * 5 + 2;
			indexBase[4] = i * 5;
			indexBase[5] = i * 5 + 3;
			indexBase[6] = i * 5;
			indexBase[7] = i * 5 + 4;
			indexBase[8] = i * 5 + 1;
			indexBase[9] = i * 5 + 2;
			indexBase[10] = i * 5 + 2;
			indexBase[11] = i * 5 + 3;
			indexBase[12] = i * 5 + 3;
			indexBase[13] = i * 5 + 4;
			indexBase[14] = i * 5 + 4;
			indexBase[15] = i * 5 + 1;

			// 5 vertex * camera.size()
			auto vertexBase = &reinterpret_cast<glm::vec3*>(vertexMem.get())[i * 5];
			for (uint8_t j = 0; j < 5; ++j)
			{
				vertexBase[j] = glm::vec3(plane[j].x, plane[j].y, plane[j].z);
			}
		}
	}

	// Volume
	{
		vector<Point3f*> corners = m_scene3d.getReconstructor().getCorners();

		Buffer::CreateInfo buffer_info;
		// Vertex Buffer
		buffer_info.Size = corners.size() * sizeof(glm::vec3);
		buffer_info.BufferType = Buffer::Type::Vertex;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "volume vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);
		// Index Buffer
		buffer_info.Size = 24 * sizeof(uint32_t);
		buffer_info.BufferType = Buffer::Type::Index;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "volume index buffer";
		auto index_buffer = Buffer::create(buffer_info);

		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.MeshTopology = Mesh::Topology::Lines;
		info.DebugName = "volume";
		m_volumeMesh = Mesh::create(info, std::move(vertex_buffer), std::move(index_buffer));
		auto vertexMem = m_volumeMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexMem = m_volumeMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexBase = reinterpret_cast<uint32_t*>(indexMem.get());
		// bottom
		indexBase[0] = 0;
		indexBase[1] = 1;
		indexBase[2] = 1;
		indexBase[3] = 2;
		indexBase[4] = 2;
		indexBase[5] = 3;
		indexBase[6] = 3;
		indexBase[7] = 0;
		// top
		indexBase[8] = 4;
		indexBase[9] = 5;
		indexBase[10] = 5;
		indexBase[11] = 6;
		indexBase[12] = 6;
		indexBase[13] = 7;
		indexBase[14] = 7;
		indexBase[15] = 4;
		// connection
		indexBase[16] = 0;
		indexBase[17] = 4;
		indexBase[18] = 1;
		indexBase[19] = 5;
		indexBase[20] = 2;
		indexBase[21] = 6;
		indexBase[22] = 3;
		indexBase[23] = 7;
		auto vertexBase = reinterpret_cast<glm::vec3*>(vertexMem.get());
		for (size_t i = 0; i < corners.size(); ++i)
		{
			vertexBase[i] = glm::vec3(corners[i]->x, corners[i]->y, corners[i]->z);
		}
	}

	// Origin
	{
		struct vertex_t
		{
		  glm::vec4 position;
		  glm::vec4 color;
		};
		// Vertex Buffer
		Buffer::CreateInfo buffer_info;
		buffer_info.Size = 6 * sizeof(vertex_t);
		buffer_info.BufferType = Buffer::Type::Vertex;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "origin vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);

		const std::vector<Mesh::MeshAttributes> attributes{
			Mesh::MeshAttributes{Mesh::MeshAttributes::DataType::Float, 4},
			Mesh::MeshAttributes{Mesh::MeshAttributes::DataType::Float, 4},
		};
		info.Attributes = attributes.data();
		info.AttributeCount = attributes.size();
		info.MeshTopology = Mesh::Topology::Lines;
		info.DebugName = "origin";
		m_wMesh = Mesh::create(info, std::move(vertex_buffer), nullptr);

		int len = m_scene3d.getSquareSideLen();
		auto x_len = static_cast<float>(len * (m_scene3d.getBoardSize().height - 1));
		auto y_len = static_cast<float>(len * (m_scene3d.getBoardSize().width - 1));
		auto z_len = static_cast<float>(len * 3);

		auto vertexMem = m_wMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto vertexBase = reinterpret_cast<vertex_t*>(vertexMem.get());
		vertexBase[0].position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vertexBase[0].color = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
		vertexBase[1].position = glm::vec4(x_len, 0.0f, 0.0f, 1.0f);
		vertexBase[1].color = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
		vertexBase[2].position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vertexBase[2].color = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f);
		vertexBase[3].position = glm::vec4(0.0f, y_len, 0.0f, 1.0f);
		vertexBase[3].color = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f);
		vertexBase[4].position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vertexBase[4].color = glm::vec4(1.0f, 0.0f, 0.0f, 0.5f);
		vertexBase[5].position = glm::vec4(0.0f, 0.0f, z_len, 1.0f);
		vertexBase[5].color = glm::vec4(1.0f, 0.0f, 0.0f, 0.5f);
	}

	// Arcball
	{
		constexpr uint8_t stack_count = 24;
		constexpr uint8_t sector_count = 48;

		Buffer::CreateInfo buffer_info;
		// Vertex Buffer
		buffer_info.Size = (stack_count + 1) * (sector_count + 1) * sizeof(glm::vec3);
		buffer_info.BufferType = Buffer::Type::Vertex;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "arcball vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);
		// Index Buffer
		buffer_info.Size = 4 * (stack_count + 1) * (sector_count + 1) * sizeof(uint32_t);
		buffer_info.BufferType = Buffer::Type::Index;
		buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
		buffer_info.DebugName = "arcball index buffer";
		auto index_buffer = Buffer::create(buffer_info);

		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.MeshTopology = Mesh::Topology::Lines;
		info.DebugName = "arcball";
		m_arcballMesh = Mesh::create(info, std::move(vertex_buffer), std::move(index_buffer));

		auto vertexMem = m_arcballMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexMem = m_arcballMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);

		constexpr float sector_step = 2 * glm::pi<float>() / sector_count;
		constexpr float stack_step = glm::pi<float>() / stack_count;

		auto vertexBase = reinterpret_cast<glm::vec3*>(vertexMem.get());
		for (uint8_t i = 0; i <= stack_count; ++i)
		{
			float stack_angle = glm::pi<float>() * 0.5f - i * stack_step;
			float xy = std::cos(stack_angle);
			float z = std::sin(stack_angle);
			for (uint8_t j = 0; j <= sector_count; ++j)
			{
				float sector_angle = j * sector_step;
				float x = xy * std::cos(sector_angle);
				float y = xy * std::sin(sector_angle);
				vertexBase[i * (sector_count + 1) + j] = glm::vec3(x, y, z);
			}
		}
		for (uint8_t i = 0; i <= stack_count; ++i)
		{
			for (uint8_t j = 0; j <= sector_count; ++j)
			{
				auto indexBase = &reinterpret_cast<uint32_t*>(indexMem.get())[4 * (i * (sector_count + 1) + j)];
				indexBase[0] = i * (sector_count + 1) + j;
				if (j == sector_count)
				{
					indexBase[1] = i * (sector_count + 1);
				}
				else
				{
					indexBase[1] = indexBase[0] + 1;
				}
				indexBase[2] = i * (sector_count + 1) + j;
				if (i == stack_count)
				{
					indexBase[3] = j;
				}
				else
				{
					indexBase[3] = indexBase[0] + (sector_count + 1);
				}
			}
		}
	}

	// Voxels using marching cubes via indirect calls through compute shader
	{
		Buffer::CreateInfo buffer_info;

		// Vertex Buffer
		// 15 possible vertex per voxel
		buffer_info.Size = sizeof(glm::vec4) * 15 * m_scene3d.getReconstructor().getVoxelCount();
		buffer_info.BufferType = Buffer::Type ::ShaderStorage;
		buffer_info.BufferUsage = Buffer::Usage::DynamicDraw;
		buffer_info.DebugName = "voxel vertex buffer";
		auto vertex_buffer = Buffer::create(buffer_info);
		const std::vector<Mesh::MeshAttributes> attributes{
			Mesh::MeshAttributes{Mesh::MeshAttributes::DataType::Float, 4},
		};
		info.Attributes = attributes.data();
		info.AttributeCount = attributes.size();
		info.MeshTopology = Mesh::Topology::Triangles;
		info.DebugName = "voxels";
		m_voxelMesh = Mesh::create(info, std::move(vertex_buffer), nullptr);

		// Voxels 3D texture
		auto dim = m_scene3d.getReconstructor().getVoxelDimension();
		Texture::CreateInfo texture_info;
		texture_info.Width = dim[0];
		texture_info.Height = dim[1];
		texture_info.Depth = dim[2];
		texture_info.DataFormat = Texture::Format::r32f;
		texture_info.DebugName = "Scalar field of voxels";
		m_scalarField = Texture::create(texture_info);

		buffer_info.Size = sizeof(DrawArraysIndirectCommand);
		buffer_info.BufferType = Buffer::Type ::IndirectDraw;
		buffer_info.BufferUsage = Buffer::Usage::DynamicCopy;
		buffer_info.DebugName = "Marching cubes indirect draw";
		m_indirectBuffer = Buffer::create(buffer_info);

		buffer_info.Size = sizeof(marchingCubeEdgeTable);
		buffer_info.BufferType = Buffer::Type ::ShaderStorage;
		buffer_info.BufferUsage = Buffer::Usage::StaticRead;
		buffer_info.DebugName = "Marching cubes edge look-up tables";
		m_marchingCubeEdgeLookUpBuffer = Buffer::create(buffer_info);
		{
			auto mem = m_marchingCubeEdgeLookUpBuffer->map(Buffer::MemoryMapAccess::Write);
			memcpy(mem.get(), marchingCubeEdgeTable, sizeof(marchingCubeEdgeTable));
		}

		buffer_info.Size = sizeof(marchingCubeTriTable);
		buffer_info.BufferType = Buffer::Type ::ShaderStorage;
		buffer_info.BufferUsage = Buffer::Usage::StaticRead;
		buffer_info.DebugName = "Marching cubes triangle look-up tables";
		m_marchingCubeTriangleLookUpBuffer = Buffer::create(buffer_info);
		{
			auto mem = m_marchingCubeTriangleLookUpBuffer->map(Buffer::MemoryMapAccess::Write);
			memcpy(mem.get(), marchingCubeTriTable, sizeof(marchingCubeTriTable));
		}
	}
}

void Renderer::initializePipelines()
{
	Pipeline::GraphicsCreateInfo info;
	info.ViewportWidth = m_scene3d.getWidth();
	info.ViewportHeight = m_scene3d.getHeight();
	info.VertexShaderSource = viewProjVertexShaderSource;
	info.FragmentShaderSource = gridFragmentShaderSource;
	info.LineWidth = 1.0f;
	info.DebugName = "grid";
	m_wireframePipeline = Pipeline::create(info);

	info.VertexShaderSource = wVertexShaderSource;
	info.FragmentShaderSource = wFragmentShaderSource;
	info.LineWidth = 1.5f;
	info.DebugName = "origin";
	m_wPipeline = Pipeline::create(info);

	info.VertexShaderSource = arcballVertexShaderSource;
	info.FragmentShaderSource = arcballFragmentShaderSource;
	info.LineWidth = 1.0f;
	info.DebugName = "arcball";
	m_arcballPipeline = Pipeline::create(info);

	Pipeline::ComputeCreateInfo compute_info;
	compute_info.ShaderSource = marchingCubeComputeShaderSource;
	compute_info.DebugName = "marching cubes";
	m_marchingCubesPipeline = Pipeline::create(compute_info);

	info.VertexShaderSource = voxelVertexShaderSource;
	info.FragmentShaderSource = voxelFragmentShaderSource;
	info.DebugName = "voxel";
	m_voxelPipeline = Pipeline::create(info);
}

void Renderer::reset()
{
	m_arc_ball.set_distance(-12500);

	m_projectionMatrix = glm::perspective(glm::radians(50.0f), m_renderer->getAspectRatio(), 1.0f, 40000.0f);
	m_viewMatrix = glm::lookAt(m_scene3d.getArcballEye(), m_scene3d.getArcballCentre(), m_scene3d.getArcballUp());

	// set up the ArcBall using the current projection matrix
	m_arc_ball.set_zoom(m_scene3d.getSphereRadius(), m_scene3d.getArcballEye(), m_scene3d.getArcballUp());
	glm::ivec4 viewport = glm::ivec4(0, 0, 0, 0);
	m_renderer->getSize(viewport.z, viewport.w);
	m_arc_ball.set_properties(m_projectionMatrix * m_viewMatrix, viewport);
}

void Renderer::quit()
{
	m_scene3d.setQuit(true);
	exit(EXIT_SUCCESS);
}

/**
 * Handle all keyboard input
 */
void Renderer::keyboard(const SDL_KeyboardEvent& event)
{
	switch (event.keysym.sym)
	{
	case SDLK_q:
		m_scene3d.setQuit(true);
		break;
	case SDLK_p:
		m_scene3d.setPaused(!m_scene3d.isPaused());
		break;
	case SDLK_b:
		m_scene3d.setCurrentFrame(m_scene3d.getCurrentFrame() - 1);
		break;
	case SDLK_n:
		m_scene3d.setCurrentFrame(m_scene3d.getCurrentFrame() + 1);
		break;
	case SDLK_r:
		m_scene3d.setRotate(!m_scene3d.isRotate());
		break;
	case SDLK_s:
		m_scene3d.setShowArcball(!m_scene3d.isShowArcball());
		break;
	case SDLK_v:
		m_scene3d.setShowVolume(!m_scene3d.isShowVolume());
		break;
	case SDLK_g:
		m_scene3d.setShowGrdFlr(!m_scene3d.isShowGrdFlr());
		break;
	case SDLK_c:
		m_scene3d.setShowCam(!m_scene3d.isShowCam());
		break;
	case SDLK_i:
		m_scene3d.setShowInfo(!m_scene3d.isShowInfo());
		break;
	case SDLK_o:
		m_scene3d.setShowOrg(!m_scene3d.isShowOrg());
		break;
	case SDLK_t:
	{
		m_scene3d.setTopView();
		m_arc_ball.reset();
		reset();
	} break;
	case SDLK_e:
		m_scene3d.calibThresholds();
		break;
	case SDLK_1:
	case SDLK_2:
	case SDLK_3:
	case SDLK_4:
	case SDLK_5:
	case SDLK_6:
	case SDLK_7:
	case SDLK_8:
	case SDLK_9:
	{
		uint32_t index = event.keysym.sym - SDLK_1;
		if (index < (int) m_scene3d.getCameras().size()) {
			m_scene3d.setCamera(index);
			m_arc_ball.reset();
			reset();
		}
	} break;
	}
}

/**
 * Handle linux mouse input (clicks and scrolls)
 */
void Renderer::mouse(const SDL_Event& event, bool& mouse_down)
{
	switch (event.type)
	{
	case SDL_MOUSEWHEEL:
		if (!m_scene3d.isCameraView())
		{
			m_arc_ball.add_distance(event.wheel.y * 250.0f);
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		m_arc_ball.start({event.motion.x, m_scene3d.getHeight() - event.motion.y - 1});
		mouse_down = true;
		break;
	case SDL_MOUSEBUTTONUP:
		mouse_down = false;
		break;
	case SDL_MOUSEMOTION:
		if (mouse_down)
		{
			int invert_y = (m_scene3d.getHeight() - event.motion.y) - 1;
			m_arc_ball.move({event.motion.x, invert_y});
		}
		break;
	}

}

void Renderer::reshape(int width, int height)
{
	float ar = (float) width / (float) height;
	m_scene3d.setSize(width, height, ar);
	initializePipelines();
	reset();
}

/**
 * Render the 3D scene
 */
void Renderer::display()
{
	auto matrix = m_arc_ball.get_matrix();
	// translate zoom on eye z
	auto zoom = glm::translate(glm::mat4(1), glm::vec3(0, 0, m_arc_ball.get_distance()));
	// rotate around arcball z
	auto rot_z = glm::eulerAngleZ(glm::radians(m_arc_ball.get_z_rotation()));
	m_viewMatrix = rot_z * zoom * matrix;

	// Compute point cloud mesh
	auto dim = m_scene3d.getReconstructor().getVoxelDimension();
	m_marchingCubesPipeline->bind();
	m_marchingCubesPipeline->setUniform("resolution", glm::vec3(dim[0], dim[1], dim[2]));
	m_marchingCubesPipeline->setUniform("indirect_data", 0, *m_indirectBuffer);
	m_marchingCubesPipeline->setUniform("edge_lut", 1, *m_marchingCubeEdgeLookUpBuffer);
	m_marchingCubesPipeline->setUniform("triangle_lut", 2, *m_marchingCubeTriangleLookUpBuffer);
	m_marchingCubesPipeline->setUniform("vertex_data", 3, m_voxelMesh->getVertexBuffer());
	m_renderer->dispatch(dim[0] / 8, dim[1] / 8, dim[2] / 8);

	m_renderPass->bind();

	m_wireframePipeline->bind();
	m_wireframePipeline->setUniform("view", m_viewMatrix);
	m_wireframePipeline->setUniform("proj", m_projectionMatrix);
	if (m_scene3d.isShowGrdFlr())
		m_gridMesh->draw();
	if (m_scene3d.isShowCam())
		m_cameraMesh->draw();
	if (m_scene3d.isShowVolume())
		m_volumeMesh->draw();
	if (m_scene3d.isShowArcball())
	{
		m_arcballPipeline->bind();
		m_arcballPipeline->setUniform("view", m_viewMatrix);
		m_arcballPipeline->setUniform("proj", m_projectionMatrix);
		m_arcballPipeline->setUniform("scale", m_scene3d.getSphereRadius());
		m_arcballMesh->draw();
	}

	auto offset = m_scene3d.getReconstructor().getOffset();
	m_voxelPipeline->bind();
	m_voxelPipeline->setUniform("view", m_viewMatrix);
	m_voxelPipeline->setUniform("proj", m_projectionMatrix);
	m_voxelPipeline->setUniform("scale", (float)m_scene3d.getReconstructor().getVoxelSize());
	m_voxelPipeline->setUniform("offset", glm::vec3(offset[0], offset[1], offset[2]));
	m_voxelMesh->draw(*m_indirectBuffer);

	if (m_scene3d.isShowOrg())
	{
		m_wPipeline->bind();
		m_wPipeline->setUniform("view", m_viewMatrix);
		m_wPipeline->setUniform("proj", m_projectionMatrix);
		m_wMesh->draw();
	}
	if (m_scene3d.isShowInfo())
	{
		std::vector<std::pair<glm::vec2, std::string>> labels;
		vector<Camera*> cameras = m_scene3d.getCameras();
		for (size_t c = 0; c < cameras.size(); ++c)
		{
			glm::ivec4 viewport = glm::ivec4(0, 0, 0, 0);
			m_renderer->getSize(viewport.z, viewport.w);
			glm::vec3 position(cameras[c]->getCameraLocation().x, cameras[c]->getCameraLocation().y, cameras[c]->getCameraLocation().z);
			auto screen_pos = glm::project(position, m_viewMatrix, m_projectionMatrix, viewport);
			labels.emplace_back(glm::vec2(screen_pos.x, viewport.w - screen_pos.y), std::to_string(c + 1));
		}
		m_ui->draw(m_renderer->getNativeWindowHandle(), labels);
		// drawInfo();
	}

	m_renderer->swapBuffers();
}

/**
 * - Update the scene with a new frame from the video
 * - Handle the keyboard input from the OpenCV window
 * - Update the OpenCV video window and frames slider position
 */
void Renderer::update()
{
	// Update the opencv image
	waitKey(10);

	Scene3DRenderer& scene3d = m_scene3d;
	if (scene3d.isQuit())
	{
		// Quit signaled
		quit();
	}
	if (scene3d.getCurrentFrame() > scene3d.getNumberOfFrames() - 2)
	{
		// Go to the start of the video if we've moved beyond the end
		scene3d.setCurrentFrame(0);
		for (size_t c = 0; c < scene3d.getCameras().size(); ++c)
			scene3d.getCameras()[c]->setVideoFrame(scene3d.getCurrentFrame());
	}
	if (scene3d.getCurrentFrame() < 0)
	{
		// Go to the end of the video if we've moved before the start
		scene3d.setCurrentFrame(scene3d.getNumberOfFrames() - 2);
		for (size_t c = 0; c < scene3d.getCameras().size(); ++c)
			scene3d.getCameras()[c]->setVideoFrame(scene3d.getCurrentFrame());
	}
	if (!scene3d.isPaused())
	{
		// If not paused move to the next frame
		scene3d.setCurrentFrame(scene3d.getCurrentFrame() + 1);
	}
	if (scene3d.getCurrentFrame() != scene3d.getPreviousFrame())
	{
		// If the current frame is different from the last iteration update stuff
		scene3d.processFrame();
		scene3d.getReconstructor().update();
		scene3d.setPreviousFrame(scene3d.getCurrentFrame());
	}
	else if (scene3d.getHThreshold() != scene3d.getPHThreshold() || scene3d.getSThreshold() != scene3d.getPSThreshold()
			|| scene3d.getVThreshold() != scene3d.getPVThreshold())
	{
		// Update the scene if one of the HSV sliders was moved (when the video is paused)
		scene3d.processFrame();
		scene3d.getReconstructor().update();

		scene3d.setPHThreshold(scene3d.getHThreshold());
		scene3d.setPSThreshold(scene3d.getSThreshold());
		scene3d.setPVThreshold(scene3d.getVThreshold());
	}

	// Auto rotate the scene
	if (scene3d.isRotate())
	{
		m_arc_ball.add_angle(2);
	}

	// Get the image and the foreground image (of set camera)
	Mat canvas, foreground;
	if (scene3d.getCurrentCamera() != -1)
	{
		canvas = scene3d.getCameras()[scene3d.getCurrentCamera()]->getFrame();
		foreground = scene3d.getCameras()[scene3d.getCurrentCamera()]->getForegroundImage();
	}
	else
	{
		canvas = scene3d.getCameras()[scene3d.getPreviousCamera()]->getFrame();
		foreground = scene3d.getCameras()[scene3d.getPreviousCamera()]->getForegroundImage();
	}

	// Concatenate the video frame with the foreground image (of set camera)
	if (!canvas.empty() && !foreground.empty())
	{
		Mat fg_im_3c;
		cvtColor(foreground, fg_im_3c, CV_GRAY2BGR);
		hconcat(canvas, fg_im_3c, canvas);
		imshow(VIDEO_WINDOW, canvas);
	}
	else if (!canvas.empty())
	{
		imshow(VIDEO_WINDOW, canvas);
	}

	// Update the frame slider position
	setTrackbarPos("Frame", VIDEO_WINDOW, scene3d.getCurrentFrame());

	auto& scalarField = scene3d.getReconstructor().getScalarField();
	m_scalarField->upload(scalarField.data(), scalarField.size() * sizeof(scalarField[0]));
}

} /* namespace nl_uu_science_gmt */

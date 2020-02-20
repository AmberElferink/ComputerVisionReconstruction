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
#include "../graphics/IndexedMesh.h"
#include "../graphics/Pipeline.h"
#include "../graphics/RenderPass.h"
#include "../graphics/Ui.h"
#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

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

constexpr std::string_view voxelVertexShaderSource =
	"#version 450 core\n"
	"layout (location = 0) in vec4 position;\n"
	"uniform mat4 view;\n"
	"uniform mat4 proj;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = proj * view * position;\n"
	"    gl_PointSize = 2.0f;\n"
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
	IndexedMesh::CreateInfo info = {};
	const std::vector<IndexedMesh::MeshAttributes> wireframe_attributes{
		IndexedMesh::MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 3},
	};

	// Grid
	{
		vector<vector<Point3i *> > floor_grid = m_scene3d.getFloorGrid();
		int gSize = m_scene3d.getNum()*2 + 1;
		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.IndexBufferSize = 4 * gSize * sizeof(uint32_t);
		info.VertexBufferSize = 4 * gSize * sizeof(glm::vec3);
		info.MeshTopology = IndexedMesh::Topology::Lines;
		info.DebugName = "grid";
		m_gridMesh = IndexedMesh::create(info);

		auto vertexMem = m_gridMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexMem = m_gridMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);
		for (int g = 0; g < gSize; g++)
		{
			auto indexBase = &reinterpret_cast<uint32_t*>(indexMem.get())[g * 4];
			indexBase[0] = g * 4;
			indexBase[1] = g * 4 + 1;
			indexBase[2] = g * 4 + 2;
			indexBase[3] = g * 4 + 3;

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
		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.IndexBufferSize = 16 * cameras.size() * sizeof(uint32_t);
		info.VertexBufferSize = 5 * cameras.size() * sizeof(glm::vec3);
		info.MeshTopology = IndexedMesh::Topology::Lines;
		info.DebugName = "cameras";
		m_cameraMesh = IndexedMesh::create(info);
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
		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.IndexBufferSize = 24 * sizeof(uint32_t);
		info.VertexBufferSize = corners.size() * sizeof(glm::vec3);
		info.MeshTopology = IndexedMesh::Topology::Lines;
		info.DebugName = "volume";
		m_volumeMesh = IndexedMesh::create(info);
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
		const std::vector<IndexedMesh::MeshAttributes> attributes{
			IndexedMesh::MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 4},
			IndexedMesh::MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 4},
		};
		struct vertex_t
		{
		  glm::vec4 position;
		  glm::vec4 color;
		};
		info.Attributes = attributes.data();
		info.AttributeCount = attributes.size();
		info.IndexBufferSize = 6 * sizeof(uint32_t);
		info.VertexBufferSize = 6 * sizeof(vertex_t);
		info.MeshTopology = IndexedMesh::Topology::Lines;
		info.DebugName = "origin";
		m_wMesh = IndexedMesh::create(info);

		auto vertexMem = m_wMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexMem = m_wMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);

		int len = m_scene3d.getSquareSideLen();
		auto x_len = static_cast<float>(len * (m_scene3d.getBoardSize().height - 1));
		auto y_len = static_cast<float>(len * (m_scene3d.getBoardSize().width - 1));
		auto z_len = static_cast<float>(len * 3);

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

		auto indexBase = reinterpret_cast<uint32_t*>(indexMem.get());
		for (size_t i = 0; i < 6; ++i)
		{
			indexBase[i] = i;
		}
	}

	// Arcball
	{
		constexpr uint8_t stack_count = 24;
		constexpr uint8_t sector_count = 48;
		info.Attributes = wireframe_attributes.data();
		info.AttributeCount = wireframe_attributes.size();
		info.IndexBufferSize = 4 * (stack_count + 1) * (sector_count + 1) * sizeof(uint32_t);
		info.VertexBufferSize = (stack_count + 1) * (sector_count + 1) * sizeof(glm::vec3);
		info.MeshTopology = IndexedMesh::Topology::Lines;
		info.DebugName = "arcball";
		m_arcballMesh = IndexedMesh::create(info);

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

	// Voxels
	{
		auto voxel_count = m_scene3d.getReconstructor().getVoxelCount();
		const std::vector<IndexedMesh::MeshAttributes> attributes{
			IndexedMesh::MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 4},
		};
		info.Attributes = attributes.data();
		info.AttributeCount = attributes.size();
		info.IndexBufferSize = voxel_count * sizeof(uint32_t);
		info.VertexBufferSize = voxel_count * sizeof(glm::vec4);
		info.DynamicVertices = true;  // Data will be uploaded at every frame
		info.MeshTopology = IndexedMesh::Topology::Points;
		info.DebugName = "voxels";
		m_voxelMesh = IndexedMesh::create(info);
		auto indexMem = m_voxelMesh->getIndexBuffer().map(Buffer::MemoryMapAccess::Write);
		auto indexBase = reinterpret_cast<uint32_t*>(indexMem.get());
		for (size_t i = 0; i < voxel_count; ++i)
		{
			indexBase[i] = i;
		}
	}
}

void Renderer::initializePipelines()
{
	Pipeline::CreateInfo info;
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

	{
		m_voxelPipeline->bind();
		m_voxelPipeline->setUniform("view", m_viewMatrix);
		m_voxelPipeline->setUniform("proj", m_projectionMatrix);
		m_voxelMesh->draw(m_scene3d.getReconstructor().getVisibleVoxels().size());
	}

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

	{
		auto mem = m_voxelMesh->getVertexBuffer().map(Buffer::MemoryMapAccess::Write);
		vector<Reconstructor::Voxel*> voxels = m_scene3d.getReconstructor().getVisibleVoxels();
		auto gpu_voxels = reinterpret_cast<glm::vec4*>(mem.get());
		for (uint32_t i = 0; i < voxels.size(); ++i)
		{
			gpu_voxels[i] = glm::vec4(voxels[i]->x, voxels[i]->y, voxels[i]->z, 1.0f);
		}
	}
}

} /* namespace nl_uu_science_gmt */

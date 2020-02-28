/*
 * Glut.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef GLUT_H_
#define GLUT_H_

#include <memory>

#include "ArcBall.h"

class Buffer;
class Context;
class Pipeline;
class Mesh;
class RenderPass;
class Texture;
class Ui;
union SDL_Event;
struct SDL_KeyboardEvent;

namespace nl_uu_science_gmt
{

class Scene3DRenderer;

class Renderer
{
public:
	explicit Renderer(Scene3DRenderer &);
	virtual ~Renderer();

	void initialize(const char* title, int argc, char* argv[]);
	void initializeGeometry();
	void initializePipelines();
	void mouse(const SDL_Event& event, bool& mouse_down);
	void keyboard(const SDL_KeyboardEvent& event);
	void reshape(int width, int height);
	void reset();
	void display();
	void update();
	void quit();

private:
  Scene3DRenderer &m_scene3d;
  ArcBall m_arc_ball;
  std::unique_ptr<Context> m_renderer;
  std::unique_ptr<Ui> m_ui;
  std::unique_ptr<RenderPass> m_renderPass;
  std::unique_ptr<RenderPass> m_overlayRenderPass;
  std::unique_ptr<Pipeline> m_wireframePipeline;
  std::unique_ptr<Pipeline> m_wPipeline;
  std::unique_ptr<Pipeline> m_arcballPipeline;
  std::unique_ptr<Pipeline> m_marchingCubesPipeline;
  std::unique_ptr<Pipeline> m_voxelPipeline;
  std::unique_ptr<Mesh> m_gridMesh;
  std::unique_ptr<Mesh> m_cameraMesh;
  std::unique_ptr<Mesh> m_volumeMesh;
  std::unique_ptr<Mesh> m_wMesh;
  std::unique_ptr<Mesh> m_arcballMesh;
  std::unique_ptr<Buffer> m_indirectBuffer;
  std::unique_ptr<Buffer> m_marchingCubeEdgeLookUpBuffer;
  std::unique_ptr<Buffer> m_marchingCubeTriangleLookUpBuffer;
  std::unique_ptr<Texture> m_scalarField;
  std::unique_ptr<Mesh> m_voxelMesh;
  glm::mat4 m_viewMatrix;
  glm::mat4 m_projectionMatrix;
};

} /* namespace nl_uu_science_gmt */

#endif /* GLUT_H_ */

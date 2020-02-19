#ifndef ARCBALL_H
#define ARCBALL_H

/* Arcball, written by Bradley Smith, March 24, 2006
 * arcball.h is free to use and modify for any purpose, with no
 * restrictions of copyright or license.
 *
 * Using the arcball:
 *   Call arcball_setzoom after setting up the projection matrix.
 *
 *     The arcball, by default, will act as if a sphere with the given
 *     radius, centred on the origin, can be directly manipulated with
 *     the mouse. Clicking on a point should drag that point to rest under
 *     the current mouse position. eye is the position of the eye relative
 *     to the origin. up is unused.
 *
 *     Alternatively, pass the value: (-radius/|eye|)
 *     This puts the arcball in a mode where the distance the mouse moves
 *     is equivalent to rotation along the axes. This acts much like a
 *     trackball. (It is for this mode that the up vector is required,
 *     which must be a unit vector.)
 *
 *     You should call arcball_setzoom after use of gluLookAt.
 *     gluLookAt(eye.x,eye.y,eye.z, ?,?,?, up.x,up.y,up.z);
 *     The arcball derives its transformation information from the
 *     openGL projection and viewport matrices. (modelview is ignored)
 *
 *     If looking at a point different from the origin, the arcball will still
 *     act as if it centred at (0,0,0). (You can use this to translate
 *     the arcball to some other part of the screen.)
 *
 *   Call arcball_start with a mouse position, and the arcball will
 *     be ready to manipulate. (Call on mouse button down.)
 *   Call arcball_move with a mouse position, and the arcball will
 *     find the rotation necessary to move the start mouse position to
 *     the current mouse position on the sphere. (Call on mouse move.)
 *   Call arcball_rotate after resetting the modelview matrix in your
 *     drawing code. It will call glRotate with its current rotation.
 *   Call arcball_reset if you wish to reset the arcball rotation.
 */

#include <glm/fwd.hpp>
#include <glm/mat4x4.hpp>

class ArcBall
{
public:
	ArcBall();
	void set_zoom(float radius, const glm::vec3& eye, const glm::vec3& up);
	void add_angle(float delta);
	void add_distance(float delta);
	/// reset the arc ball
	void reset();
	/// begin arc ball rotation
	void start(const glm::ivec2& position);
	/// update current arc ball rotation
	void move(const glm::ivec2& position);

	void set_properties(const glm::mat4& proj, const glm::uvec4& view);
	float get_distance() const;
	glm::mat4 get_matrix() const;
	float get_z_rotation() const;

private:
	/// find the intersection with the plane through the visible edge
	glm::vec3 edge_coords(const glm::vec3& m);
	/// find the intersection with the sphere
	glm::vec3 sphere_coords(const glm::vec2& position);
	/// get intersection with plane for "trackball" style rotation
	glm::vec3 planar_coords(const glm::vec2& position);

	glm::mat4 m_quat;
	glm::mat4 m_last;
	glm::mat4 m_next;
	/// the distance from the origin to the eye
	float m_zoom;
	float m_zoom2;
	/// the radius of the arc ball
	float m_sphere;
	float m_sphere2;
	/// the distance from the origin of the plane that intersects
	/// the edge of the visible sphere (tangent to a ray from the eye)
	float m_edge;
	/// whether we are using a sphere or plane
	bool m_planar;
	float m_plane_dist;
	float m_rotation;
	float m_distance;
	glm::vec3 m_start;
	glm::vec3 m_curr;
	glm::vec3 m_eye;
	glm::vec3 m_eye_dir;
	glm::vec3 m_up;
	glm::vec3 m_out;
	glm::mat4 m_glp;
	glm::mat4 m_glm;
	glm::uvec4 m_glv;
};

#endif

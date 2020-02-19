/* Arcball, written by Bradley Smith, March 24, 2006
 * arcball.cpp is free to use and modify for any purpose, with no
 * restrictions of copyright or license.
 *
 * See arcball.h for usage details.
 */

#include "ArcBall.h"

#include <glm/geometric.hpp>
#include <glm/gtx/quaternion.hpp>

ArcBall::ArcBall()
	: m_quat(1.0f)
	, m_last(1.0f)
	, m_next(1.0f)
	, m_zoom(1.0f)
	, m_zoom2(1.0f)
	, m_sphere(1.0f)
	, m_sphere2(1.0f)
	, m_edge(1.0f)
	, m_planar(false)
	, m_plane_dist(0.5f)
	, m_rotation(0)
	, m_distance(0)
	, m_start(0, 0, 1)
	, m_curr(0, 0, 1)
	, m_eye(0, 0, 1)
	, m_eye_dir(0, 0, 1)
	, m_up(0, 1, 0)
	, m_out(1, 0, 0)
	, m_glp(1.0)
	, m_glm(1.0f)
	, m_glv(0, 0, 1024, 768)
{
}

void ArcBall::set_zoom(float radius, const glm::vec3& eye, const glm::vec3& up)
{
	m_eye = eye; // store eye vector
	m_zoom2 = glm::dot(m_eye, m_eye);
	m_zoom = glm::sqrt(m_zoom2); // store eye distance
	m_sphere = radius; // sphere radius
	m_sphere2 = m_sphere * m_sphere;
	m_eye_dir = m_eye * (1.0f / m_zoom); // distance to eye
	m_edge = m_sphere2 / m_zoom; // plane of visible edge

	// trackball mode
	if (m_sphere <= 0.0)
	{
		m_planar = true;
		m_up = up;
		m_out = glm::cross(m_eye_dir, m_up);
		m_plane_dist = (0.0f - m_sphere) * m_zoom;
	}
	else
	{
		m_planar = false;
	}
}

void ArcBall::add_angle(float delta)
{
	m_rotation += delta;
	m_rotation = std::fmod(m_rotation, 360);
}

void ArcBall::add_distance(float delta)
{
	m_distance += delta;
}

void ArcBall::set_distance(float distance)
{
	m_distance = distance;
}

glm::vec3 ArcBall::edge_coords(const glm::vec3& m)
{
	// find the intersection of the edge plane and the ray
	float t = (m_edge - m_zoom) / glm::dot(m_eye_dir, m);
	glm::vec3 a = m_eye + (m * t);
	// find the direction of the eye-axis from that point
	// along the edge plane
	glm::vec3 c = (m_eye_dir * m_edge) - a;

	// find the intersection of the sphere with the ray going from
	// the plane outside the sphere toward the eye-axis.
	float ac = glm::dot(a, c);
	float c2 = glm::dot(c, c);
	float q = (0.0f - ac - glm::sqrt(ac * ac - c2 * (glm::dot(a, a) - m_sphere2))) / c2;

	return glm::normalize(a + (c * q));
}

glm::vec3 ArcBall::sphere_coords(const glm::vec2& position)
{
	glm::vec3 p = glm::unProject(glm::vec3(position, 0), m_glm, m_glp, m_glv);
	glm::vec3 m = p - m_eye;

	// mouse position represents ray: eye + t*m
	// intersecting with a sphere centered at the origin
	float a = glm::dot(m, m);
	float b = glm::dot(m_eye, m);
	float root = (b * b) - a * (m_zoom2 - m_sphere2);
	if (root <= 0)
		return edge_coords(m);
	float t = (0.0f - b - glm::sqrt(root)) / a;
	return glm::normalize(m_eye + (m * t));
}

glm::vec3 ArcBall::planar_coords(const glm::vec2& position)
{
	glm::vec3 p = glm::unProject(glm::vec3(position, 0), m_glm, m_glp, m_glv);
	glm::vec3 m = p - m_eye;

	// intersect the point with the trackball plane
	float t = (m_plane_dist - m_zoom) / glm::dot(m_eye_dir, m);
	glm::vec3 d = m_eye + m * t;

	return glm::vec3(glm::dot(d, m_up), glm::dot(d, m_out), 0.0);
}

void ArcBall::reset()
{
	m_quat = glm::mat4(1.0f);
	m_last = glm::mat4(1.0f);
	m_distance = 0;
	m_rotation = 0;
}

void ArcBall::start(const glm::ivec2& position)
{
	// saves a copy of the current rotation for comparison
	m_last = m_quat;
	if (m_planar)
		m_start = planar_coords(glm::vec2(position));
	else
		m_start = sphere_coords(glm::vec2(position));
}

void ArcBall::move(const glm::ivec2& position)
{
	if (m_planar)
	{
		m_curr = planar_coords(glm::vec2(position));
		if (glm::all(glm::equal(m_curr, m_start)))
			return;

		// d is motion since the last position
		glm::vec3 d = m_curr - m_start;

		float angle = glm::length(d) * 0.5f;
		float cosa = glm::cos(angle);
		float sina = glm::sin(angle);
		// p is perpendicular to d
		glm::vec3 p = glm::normalize((m_out * d.x) - (m_up * d.y)) * sina;

		m_next = glm::toMat4(glm::quat(cosa, p.x, p.y, p.z));
		m_quat = m_last * m_next;
		// planar style only ever relates to the last point
		m_last = m_quat;
		m_start = m_curr;

	}
	else
	{
		m_curr = sphere_coords(glm::vec2(position));

		if (glm::all(glm::equal(m_curr, m_start)))
		{ // avoid potential rare divide by tiny
			m_quat = m_last;
			return;
		}

		// use a dot product to get the angle between them
		// use a cross product to get the vector to rotate around
		float cos2a = glm::dot(m_start, m_curr);
		float sina = glm::sqrt((1.0f - cos2a) * 0.5f);
		float cosa = glm::sqrt((1.0f + cos2a) * 0.5f);
		glm::vec3 cross = glm::normalize(glm::cross(m_start, m_curr)) * sina;

		m_next = glm::toMat4(glm::quat(cosa, cross.x, cross.y, cross.z));

		// update the rotation matrix
		m_quat = m_last * m_next;
	}
}

void ArcBall::set_properties(const glm::mat4& proj, const glm::uvec4& view)
{
	m_glp = proj;
	m_glv = view;
}

float ArcBall::get_distance() const
{
	return m_distance;
}

glm::mat4 ArcBall::get_matrix() const
{
	return m_quat;
}

float ArcBall::get_z_rotation() const
{
	return m_rotation;
}

#include "Mesh.h"

#include <cassert>
#include <cstring>

#include <glad/glad.h>

#include "Buffer.h"

namespace
{

    constexpr float quad_vertices[] = {
            // ,---------- u
            // |     ,---- v
            // |     |
            0.0f, 0.0f, // 0 top left
            1.00, 0.0f, // 1 top right
            1.0f, 1.0f, // 2 bottom right
            0.0f, 1.0f, // 3 bottom left
    };
    constexpr uint32_t quad_indices[] = {0, 1, 2, 2, 3, 0};

    constexpr float axis_vertices[] = {
            // ,---------------------------------- x
            // |     ,---------------------------- y
            // |     |     ,---------------------- z
            // |     |     |
            // |     |     |     ,---------------- R
            // |     |     |     |     ,---------- G
            // |     |     |     |     |     ,---- B
            // |     |     |     |     |     |
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // 0 center
            1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // 1 x axis
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // 2 center
            0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // 3 y axis
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, // 4 center
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // 5 z axis
    };

//cube vertices from: http://www.opengl-tutorial.org/beginners-tutorials/tutorial-4-a-colored-cube/
// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
    constexpr float cube_vertices[] = {
            // Left
            0.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,  -1.0f, 0.0f, 0.0f,
            // Right
            1.0f, 1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f,

            // Top
            1.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f,
            // Bottom
            1.0f, 0.0f, 1.0f,  0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 0.0f,  0.0f, -1.0f, 0.0f,
            1.0f, 0.0f, 0.0f,  0.0f, -1.0f, 0.0f,
            1.0f, 0.0f, 1.0f,  0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,  0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 0.0f,  0.0f, -1.0f, 0.0f,

            // Front
            0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
            // Back
            1.0f, 1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, 0.0f,  0.0f, 0.0f, -1.0f,
            0.0f, 1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
            1.0f, 1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
            1.0f, 0.0f, 0.0f,  0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, 0.0f,  0.0f, 0.0f, -1.0f,
    };
} // namespace

std::unique_ptr<Mesh> Mesh::create(const CreateInfo &info, std::unique_ptr<Buffer>&& vertex_buffer, std::unique_ptr<Buffer>&& index_buffer)
{
    assert(vertex_buffer);

    uint32_t vao;
    glCreateVertexArrays(1, &vao);
    if (info.DebugName.data())
    {
        // to be able to read it in RenderDoc/errors
        glObjectLabel(
            GL_VERTEX_ARRAY, vao, -1,
            (info.DebugName.data() + std::string(" vertex array object"))
                .c_str());
    }

    uint32_t totalStride = 0;
    for (uint32_t i = 0; i < info.AttributeCount; ++i)
    {
        auto size = info.Attributes[i].Count;
        switch (info.Attributes[i].Type)
        {
            case Mesh::MeshAttributes::DataType::Float:
                size *= sizeof(float);
                break;
            default:
                // printf("unsupported type\n");
                assert(false);
                return nullptr;
        }
        totalStride += size;
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer->handle_);
    // binds buffers to the slot in the vao, and this makes no sense but is
    // needed somehow
    if (index_buffer)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer->handle_);
    }

    glVertexArrayVertexBuffer(vao, 0, vertex_buffer->handle_, 0, totalStride);

    uintptr_t attr_offset = 0;
    for (uint32_t i = 0; i < info.AttributeCount; ++i)
    {
        auto size = info.Attributes[i].Count;
        switch (info.Attributes[i].Type)
        {
            case MeshAttributes::DataType::Float:
                size *= sizeof(float);
                break;
            default:
                // printf("unsupported type\n");
                assert(false);
                return nullptr;
        }
        // tell the gpu (and RenderDoc) you use data of a specific type for the
        // vertices at a specific position in the shader
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, info.Attributes[i].Count,
                info.Attributes[i].Type + GL_BYTE, GL_FALSE, totalStride,
                reinterpret_cast<const void *>(attr_offset));
        attr_offset += size;
    }

    uint32_t element_count = index_buffer ? index_buffer->size_ / sizeof(uint32_t)
        : vertex_buffer->size_ / totalStride;

    return std::unique_ptr<Mesh>(
            new Mesh(std::move(vertex_buffer), std::move(index_buffer),
                     vao, element_count, info.MeshTopology));
}

std::unique_ptr<Mesh>
Mesh::createFullscreenQuad(const std::string_view &debug_name)
{
    Buffer::CreateInfo buffer_info;

    // Vertex Buffer
    buffer_info.Size = sizeof(quad_vertices);
    buffer_info.BufferType = Buffer::Type::Vertex;
    buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
    if (debug_name.data())
    {
        buffer_info.DebugName = debug_name.data() + std::string(" vertex buffer");
    }
    auto vertex_buffer = Buffer::create(buffer_info);
    if (!vertex_buffer)
        return nullptr;

    // Index Buffer
    buffer_info.Size = sizeof(quad_indices);
    buffer_info.BufferType = Buffer::Type::Index;
    buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
    if (debug_name.data())
    {
        buffer_info.DebugName = debug_name.data() + std::string(" index buffer");
    }
    auto index_buffer = Buffer::create(buffer_info);
    if (!index_buffer)
        return nullptr;

    const std::vector<Mesh::MeshAttributes> attributes{
            MeshAttributes{Mesh::MeshAttributes::DataType::Float, 2}};
    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.MeshTopology = Topology::Triangles;
    info.DebugName = debug_name;
    auto fullscreen_quad = Mesh::create(info, std::move(vertex_buffer), std::move(index_buffer));

    std::memcpy(fullscreen_quad->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            quad_vertices, sizeof(quad_vertices));
    std::memcpy(fullscreen_quad->getIndexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            quad_indices, sizeof(quad_indices));

    return fullscreen_quad;
}

std::unique_ptr<Mesh> Mesh::createAxis(const std::string_view &debug_name)
{
    Buffer::CreateInfo buffer_info;

    // Vertex Buffer
    buffer_info.Size = sizeof(axis_vertices);
    buffer_info.BufferType = Buffer::Type::Vertex;
    buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
    if (debug_name.data())
    {
        buffer_info.DebugName = debug_name.data() + std::string(" vertex buffer");
    }
    auto vertex_buffer = Buffer::create(buffer_info);
    if (!vertex_buffer)
        return nullptr;

    const std::vector<Mesh::MeshAttributes> attributes{
            MeshAttributes{Mesh::MeshAttributes::DataType::Float, 3}, // position
            MeshAttributes{Mesh::MeshAttributes::DataType::Float, 3}, // color
    };

    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.MeshTopology = Topology::Lines;
    info.DebugName = debug_name;
    auto axis = Mesh::create(info, std::move(vertex_buffer), nullptr);

    std::memcpy(axis->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            axis_vertices, sizeof(axis_vertices));

    return axis;
}

std::unique_ptr<Mesh> Mesh::createCube(const std::string_view &debug_name)
{
    Buffer::CreateInfo buffer_info;

    // Vertex Buffer
    buffer_info.Size = sizeof(cube_vertices);
    buffer_info.BufferType = Buffer::Type::Vertex;
    buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
    if (debug_name.data())
    {
        buffer_info.DebugName = debug_name.data() + std::string(" vertex buffer");
    }
    auto vertex_buffer = Buffer::create(buffer_info);
    if (!vertex_buffer)
        return nullptr;

    const std::vector<Mesh::MeshAttributes> attributes{
            MeshAttributes{Mesh::MeshAttributes::DataType::Float, 3}, // position
            MeshAttributes{Mesh::MeshAttributes::DataType::Float, 3}, //normals
    };
    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.MeshTopology = Topology::Triangles;
    info.DebugName = debug_name;
    auto cube = Mesh::create(info, std::move(vertex_buffer), nullptr);

    std::memcpy(cube->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            cube_vertices, sizeof(cube_vertices));

    return cube;
}

Mesh::Mesh(std::unique_ptr<Buffer>&& vertex_buffer, std::unique_ptr<Buffer>&& index_buffer,
           uint32_t vao, uint32_t element_count, Topology topology)
    : vertex_buffer_(std::move(vertex_buffer))
    , index_buffer_(std::move(index_buffer))
    , vao_(vao)
    , element_count_(element_count)
    , topology_(topology)
{
}

Mesh::~Mesh()
{
    glDeleteVertexArrays(1, &vao_);
}

void Mesh::draw() const
{
    draw(element_count_);
}

void Mesh::draw(uint32_t count) const
{
    assert(count <= element_count_);
    bind();
    if (index_buffer_)
    {
        glDrawElements(topology_, count, GL_UNSIGNED_INT, nullptr);
    }
    else
    {
        glDrawArrays(topology_, 0, count);
    }
}

void Mesh::bind() const
{
    glBindVertexArray(vao_);
}

Buffer& Mesh::getVertexBuffer() const
{
    return *vertex_buffer_;
}

Buffer& Mesh::getIndexBuffer() const
{
    assert(index_buffer_);
    return *index_buffer_;
}

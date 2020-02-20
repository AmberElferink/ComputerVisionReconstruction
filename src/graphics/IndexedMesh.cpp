#include "IndexedMesh.h"

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
    constexpr uint32_t axis_indices[] = {0, 1, 2, 3, 4, 5};

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

    constexpr uint32_t cube_indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
} // namespace

std::unique_ptr<IndexedMesh> IndexedMesh::create(const CreateInfo &info)
{
    Buffer::CreateInfo buffer_info;

    // Vertex Buffer
    buffer_info.Size = info.VertexBufferSize;
    buffer_info.BufferType = Buffer::Type::Vertex;
    buffer_info.BufferUsage = info.DynamicVertices ? Buffer::Usage::DynamicDraw : Buffer::Usage::StaticDraw;
    if (info.DebugName.data())
    {
        buffer_info.DebugName = info.DebugName.data() + std::string(" vertex buffer");
    }
    auto vertex_buffer = Buffer::create(buffer_info);
    if (!vertex_buffer)
        return nullptr;

    // Index Buffer
    buffer_info.Size = info.IndexBufferSize;
    buffer_info.BufferType = Buffer::Type::Index;
    buffer_info.BufferUsage = Buffer::Usage::StaticDraw;
    if (info.DebugName.data())
    {
        buffer_info.DebugName = info.DebugName.data() + std::string(" index buffer");
    }
    auto index_buffer = Buffer::create(buffer_info);
    if (!index_buffer)
        return nullptr;

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
            case IndexedMesh::MeshAttributes::DataType::Float:
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
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer->handle_);

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

    return std::unique_ptr<IndexedMesh>(
            new IndexedMesh(std::move(vertex_buffer), std::move(index_buffer),
                            vao, info.IndexBufferSize / sizeof(uint32_t),
                            info.MeshTopology));
}

std::unique_ptr<IndexedMesh>
IndexedMesh::createFullscreenQuad(const std::string_view &debug_name)
{
    const std::vector<IndexedMesh::MeshAttributes> attributes{
            MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 2}};
    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.VertexBufferSize = sizeof(quad_vertices);
    info.IndexBufferSize = sizeof(quad_indices);
    info.MeshTopology = Topology::Triangles;
    info.DebugName = debug_name;
    auto fullscreen_quad = IndexedMesh::create(info);

    std::memcpy(fullscreen_quad->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            quad_vertices, sizeof(quad_vertices));
    std::memcpy(fullscreen_quad->getIndexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            quad_indices, sizeof(quad_indices));

    return fullscreen_quad;
}

std::unique_ptr<IndexedMesh> IndexedMesh::createAxis(const std::string_view &debug_name)
{
    const std::vector<IndexedMesh::MeshAttributes> attributes{
            MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 3}, // position
            MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 3}, // color
    };
    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.VertexBufferSize = sizeof(axis_vertices);
    info.IndexBufferSize = sizeof(axis_indices);
    info.MeshTopology = Topology::Lines;
    info.DebugName = debug_name;
    auto axis = IndexedMesh::create(info);

    std::memcpy(axis->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            axis_vertices, sizeof(axis_vertices));
    std::memcpy(axis->getIndexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            axis_indices, sizeof(axis_indices));

    return axis;
}

std::unique_ptr<IndexedMesh> IndexedMesh::createCube(const std::string_view &debug_name)
{
    const std::vector<IndexedMesh::MeshAttributes> attributes{
            MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 3}, // position
            MeshAttributes{IndexedMesh::MeshAttributes::DataType::Float, 3}, //normals
    };
    CreateInfo info;
    info.Attributes = attributes.data();
    info.AttributeCount = attributes.size();
    info.VertexBufferSize = sizeof(cube_vertices);
    info.IndexBufferSize = sizeof(cube_indices);
    info.MeshTopology = Topology::Triangles;
    info.DebugName = debug_name;
    auto cube = IndexedMesh::create(info);

    std::memcpy(cube->getVertexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            cube_vertices, sizeof(cube_vertices));
    std::memcpy(cube->getIndexBuffer().map(Buffer::MemoryMapAccess::Write).get(),
            cube_indices, sizeof(cube_indices));

    return cube;
}

IndexedMesh::IndexedMesh(std::unique_ptr<Buffer>&& vertex_buffer, std::unique_ptr<Buffer>&& index_buffer,
                         uint32_t vao, uint32_t element_count, Topology topology)
    : vertex_buffer_(std::move(vertex_buffer))
    , index_buffer_(std::move(index_buffer))
    , vao_(vao)
    , element_count(element_count)
    , topology(topology)
{
}

IndexedMesh::~IndexedMesh()
{
    glDeleteBuffers(2, reinterpret_cast<uint32_t *>(this));
    glDeleteVertexArrays(1, &vao_);
}

void IndexedMesh::draw() const
{
    bind();
    glDrawElements(topology, element_count, GL_UNSIGNED_INT, nullptr);
}

void IndexedMesh::draw(uint32_t count) const
{
    assert(count <= element_count);
    bind();
    glDrawElements(topology, count, GL_UNSIGNED_INT, nullptr);
}

void IndexedMesh::bind() const
{
    glBindVertexArray(vao_);
}

Buffer& IndexedMesh::getVertexBuffer() const
{
    return *vertex_buffer_;
}

Buffer& IndexedMesh::getIndexBuffer() const
{
    return *index_buffer_;
}

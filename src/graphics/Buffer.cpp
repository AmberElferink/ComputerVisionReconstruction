#include "Buffer.h"

#include <cassert>
#include <array>
#include <glad/glad.h>

namespace {
std::array<uint32_t, 4> type_lookup {
    /* Vertex        */ GL_ARRAY_BUFFER,
    /* Index         */ GL_ELEMENT_ARRAY_BUFFER,
    /* ShaderStorage */ GL_SHADER_STORAGE_BUFFER,
    /* IndirectDraw  */ GL_DRAW_INDIRECT_BUFFER,
};
}

std::unique_ptr<Buffer> Buffer::create(const CreateInfo& info)
{
    uint32_t handle;
    glCreateBuffers(1, &handle); // create buffer pointer on gpu
    if (handle == 0)
    {
        return nullptr;
    }
    if (!info.DebugName.empty()) {
        // to be able to read it in RenderDoc/errors
        glObjectLabel(GL_BUFFER, handle, -1, info.DebugName.data());
    }

    uint32_t usage = static_cast<uint32_t>(info.BufferUsage) + GL_STREAM_DRAW;
    uint32_t target = type_lookup[static_cast<uint32_t>(info.BufferType)];
    glNamedBufferData(handle, info.Size, nullptr, usage);

    return std::unique_ptr<Buffer>(new Buffer(handle, target, info.Size));
}

Buffer::Buffer(uint32_t handle, uint32_t target, uint32_t size)
    : handle_(handle)
    , target_(target)
    , size_(size)
{
}

Buffer::~Buffer()
{
    glDeleteBuffers(1, &handle_);
}

std::unique_ptr<uint8_t, Buffer::MemoryUnmapper>
Buffer::map(Buffer::MemoryMapAccess access)
{
    if (access <= 0 || access > 3)
    {
        return nullptr;
    }
    void *mapped = glMapNamedBuffer(handle_, GL_READ_ONLY - 1 + access);
    return std::unique_ptr<uint8_t, MemoryUnmapper>(
        reinterpret_cast<uint8_t *>(mapped), {handle_});
}

void Buffer::bind() const
{
    glBindBuffer(target_, handle_);
}

void Buffer::bind(Type type) const
{
    uint32_t target = type_lookup[static_cast<uint32_t>(type)];
    glBindBuffer(target, handle_);
}

void Buffer::MemoryUnmapper::operator()(const uint8_t *mapped)
{
    if (mapped)
    {
        glUnmapNamedBuffer(id_);
    }
}

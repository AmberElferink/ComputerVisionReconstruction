#include "GraphicsPipeline.h"

#include <glad/glad.h>
#include <glm/fwd.hpp>

#include "../Buffer.h"

GraphicsPipeline::GraphicsPipeline(uint32_t program, uint32_t viewportWidth, uint32_t viewportHeight)
    : program_(program)
    , viewportWidth_(viewportWidth)
    , viewportHeight_(viewportHeight)
{}

GraphicsPipeline::~GraphicsPipeline() { glDeleteProgram(program_); }

void GraphicsPipeline::bind() {
    glViewport(0, 0, viewportWidth_, viewportHeight_);
    glUseProgram(program_);
}

bool GraphicsPipeline::setUniform(const std::string_view& uniform_name, const glm::mat4& uniform)
{
    auto index = glGetUniformLocation(program_, uniform_name.data());
    if (index == GL_INVALID_INDEX) {
        std::fprintf(stderr, "Could not bind uniform %s: name not present\n", uniform_name.data());
        return false;
    }
    glProgramUniformMatrix4fv(program_, index, 1, false, (float*)&uniform);

    return true;
}

bool GraphicsPipeline::setUniform(const std::string_view& uniform_name, const glm::vec4& uniform)
{
    auto index = glGetUniformLocation(program_, uniform_name.data());
    if (index == GL_INVALID_INDEX) {
        std::fprintf(stderr, "Could not bind uniform %s: name not present\n", uniform_name.data());
        return false;
    }
    glProgramUniform4fv(program_, index, 1, (float*)&uniform);

    return true;
}

bool GraphicsPipeline::setUniform(const std::string_view& uniform_name, const glm::vec3& uniform)
{
    auto index = glGetUniformLocation(program_, uniform_name.data());
    if (index == GL_INVALID_INDEX) {
        std::fprintf(stderr, "Could not bind uniform %s: name not present\n", uniform_name.data());
        return false;
    }
    glProgramUniform3fv(program_, index, 1, (float*)&uniform);

    return true;
}

bool GraphicsPipeline::setUniform(const std::string_view& uniform_name, const float& uniform)
{
    auto index = glGetUniformLocation(program_, uniform_name.data());
    if (index == GL_INVALID_INDEX) {
        std::fprintf(stderr, "Could not bind uniform %s: name not present\n", uniform_name.data());
        return false;
    }
    glProgramUniform1f(program_, index, uniform);

    return true;
}

bool GraphicsPipeline::setUniform(const std::string_view& uniform_name, uint32_t index, const Buffer& uniform)
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, uniform.handle_);

    return true;
}
